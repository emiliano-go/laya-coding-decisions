"""Frozen candidate extraction and shortlisting for failure-driven localization.

Reference implementation for both the offline probe (`localize_failure.py`) and
the Host (`dsh-totem/src/localize-candidates.ts`). The rules here are frozen:
changing a regex, the exclusion list, the text format, or a score weight
invalidates the parity fixture and must be recorded in `RESULTS.md`.

The extractor is deliberately regex-based, not AST-based, because the Host is
TypeScript and cannot parse Python. AST is used only to compute gold
(`build_failure_loc.py`); the ranking must use the same rules the product ships.

Stdlib only, no network.

    python scripts/localize_candidates.py --self-check
    python scripts/localize_candidates.py --dump /path/to/repo
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

# --- frozen rules -----------------------------------------------------------

DEF_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
CLASS_RE = re.compile(r"^(\s*)class\s+([A-Za-z_]\w*)\b")
DOC_RE = re.compile(r"^\s*[rRbBuUfF]{0,2}(\"\"\"|'''|\"|')")

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "build", "dist", "docs",
    "node_modules", "vendor", "site-packages", ".tox", ".mypy_cache",
    ".ruff_cache", "__pycache__", ".pytest_cache", ".eggs", ".idea", ".vscode",
    "third_party", "external",
}
TEST_DIRS = {"tests", "test", "testing", "spec"}

MAX_FILES = 3000
MAX_SECONDS = 5.0
MAX_BYTES = 512 * 1024
MAX_TEXT = 160
MAX_TASK = 400
MAX_CANDIDATES = 16

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

FRAME_LONG_RE = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>[^\s]+))?'
)
FRAME_SHORT_RE = re.compile(
    r"^(?P<file>[\w./\\-]+\.py):(?P<line>\d+)(?::\s*(?:in\s+(?P<func>[\w.]+))?)?"
)
FAILED_RE = re.compile(r"^FAILED\s+(?P<nodeid>\S+?)(?:\s+-\s+.*)?$")
ASSERT_RE = re.compile(r"^\s*E\s+(?P<detail>.+)$")

INSTRUCTIONS = "A test is failing. Which function most likely contains the bug?"


# --- helpers ----------------------------------------------------------------

def indent_of(line: str) -> int:
    """Leading indentation in columns, tabs counted as four."""
    width = 0
    for char in line:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4
        else:
            break
    return width


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def strip_root(failure: str, root: str = "") -> str:
    """Rewrite absolute repo paths in a traceback to repo-relative ones.

    Only a path *start* is rewritten. A blind `str.replace("/testbed/", "")`
    corrupts nested paths such as `/testbed/opt/miniconda3/envs/testbed/lib`,
    which then look like files from the repository.
    """
    text = failure or ""
    prefixes = []
    if root:
        prefixes.append(os.path.abspath(root).rstrip("/") + "/")
    prefixes.extend(["/testbed/", "/repo/", "/workspace/"])
    for prefix in prefixes:
        text = re.sub(r"(?<![\w/])" + re.escape(prefix), "", text)
    return text


def is_test_path(rel: str) -> bool:
    parts = [part.lower() for part in rel.replace("\\", "/").split("/")]
    if any(part in TEST_DIRS for part in parts[:-1]):
        return True
    name = parts[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name in ("conftest.py", "setup.py", "noxfile.py")
    )


# --- traceback parsing ------------------------------------------------------

EXTERNAL_RE = re.compile(r"(^|/)(site-packages|dist-packages)/")
EXTERNAL_PREFIXES = ("opt/", "usr/", "lib/", "lib64/", "bin/", "etc/")


def is_external_path(rel: str) -> bool:
    """True for interpreter/plugin frames that cannot name a repo candidate."""
    text = rel.replace("\\", "/").lstrip("./")
    return text.startswith(EXTERNAL_PREFIXES) or EXTERNAL_RE.search(text) is not None


def parse_frames(failure: str, root: str = "") -> list[dict]:
    """Ordered traceback frames as {file, line, func}; repeats are collapsed.

    Third-party frames (pytest plugins, site-packages) are dropped: they cannot
    be candidates and they otherwise win the shortlist on uninteresting files.
    """
    text = strip_root(failure, root)
    frames: list[dict] = []
    for raw in text.splitlines():
        match = FRAME_LONG_RE.search(raw) or FRAME_SHORT_RE.match(raw.strip())
        if match is None:
            continue
        file = match.group("file").lstrip("./")
        if is_external_path(file):
            continue
        frame = {
            "file": file,
            "line": int(match.group("line")),
            "func": match.group("func") or "",
        }
        if frames and frames[-1] == frame:
            continue
        frames.append(frame)
    return frames


def parse_failed_tests(failure: str) -> list[str]:
    """`FAILED nodeid` entries, in order."""
    out = []
    for raw in (failure or "").splitlines():
        match = FAILED_RE.match(raw.strip())
        if match is not None:
            out.append(match.group("nodeid"))
    return out


SECTION_RE = re.compile(r"^=+\s+(?P<name>[A-Za-z][A-Za-z ]*?)\s+=+$")
BLOCK_RE = re.compile(r"^_+\s*(?P<name>\S.*?)\s*_+$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean_stdout(stdout: str) -> str:
    """Strip ANSI escapes so the same text tokenises identically everywhere."""
    return ANSI_RE.sub("", stdout or "")


def focus_failure(stdout: str, limit: int = 8000) -> str:
    """Keep the failing node ids, the first failure blocks, and the summary.

    A whole pytest run cannot be truncated from one end: a plugin that raises at
    teardown leaves its own traceback at the very end, while the `path:line`
    marker that identifies the failing frame sits at the *end* of each failure
    block, after a long locals dump. Each block therefore keeps its head (the
    assertion and the traced source) and its tail (the frame marker).
    """
    text = ANSI_RE.sub("", stdout or "")
    lines = text.splitlines()
    nodeids = [line.strip() for line in lines if line.strip().startswith("FAILED ")]
    sections: list[str] = []
    collecting = False
    for line in lines:
        match = SECTION_RE.match(line.strip())
        if match is not None:
            name = match.group("name").strip().lower()
            collecting = name in ("failures", "errors")
            if collecting:
                sections.append(line)
            continue
        if collecting:
            sections.append(line)
    blocks: list[str] = []
    current: list[str] | None = None
    for line in sections:
        if BLOCK_RE.match(line.strip()) is not None:
            if current:
                blocks.append("\n".join(current).strip())
            current = [line]
        elif current is not None:
            current.append(line)
        elif line.strip():
            blocks.append(line.strip())
    if current:
        blocks.append("\n".join(current).strip())
    summary: list[str] = []
    in_summary = False
    for line in lines:
        match = SECTION_RE.match(line.strip())
        if match is not None:
            in_summary = match.group("name").strip().lower().startswith("short test summary")
            if in_summary:
                summary.append(line)
            continue
        if in_summary:
            summary.append(line)

    head = ("Failing tests:\n" + "\n".join(nodeids[:12])) if nodeids else ""
    keep: list[str] = []
    for block in blocks[:2]:
        if len(block) <= 3500:
            keep.append(block)
        else:
            keep.append(block[:2600] + "\n...\n" + block[-900:])
    body = "\n\n".join(keep).strip()
    tail = "\n".join(summary).strip()
    if not body and not tail:
        body = text.strip()
    reserve = min(1200, max(0, limit // 4))
    head_text = head[: max(0, limit - reserve)]
    body_text = body[: max(0, limit - len(head_text) - reserve)]
    tail_text = tail[-reserve:] if reserve else ""
    out = "\n\n".join(part for part in (head_text, body_text, tail_text) if part)
    return out[:limit] if limit > 0 else out


def query_text(failure: str) -> str:
    """Failure text reduced to what should drive retrieval.

    Keeps the traceback, the FAILED node ids, and the `E   ...` assertion
    details; drops pytest progress dots, blank lines, and cache banners.
    """
    kept = []
    for raw in (failure or "").splitlines():
        line = raw.strip()
        if not line or set(line) <= {".", "s", "F", "E", "x", "X"}:
            continue
        if line.startswith("=") and "passed" in line:
            continue
        kept.append(line)
    return "\n".join(kept)


# --- repo walk and extraction ----------------------------------------------

def _read_text(path: str) -> str:
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return ""
        with open(path, errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def walk_repo(root: str, *, max_files: int = MAX_FILES, max_seconds: float = MAX_SECONDS) -> list[dict]:
    """Deterministic, bounded list of {path, text} for source `.py` files."""
    started = time.monotonic()
    found: list[dict] = []
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if is_test_path(rel):
                continue
            if len(found) >= max_files or time.monotonic() - started > max_seconds:
                return found
            text = _read_text(path)
            if text:
                found.append({"path": rel, "text": text})
    return found


def _signature(lines: list[str], index: int) -> tuple[str, int]:
    """(args, body_index) for the `def` at `lines[index]`, honouring wrapping."""
    # Comments inside a wrapped signature would otherwise be read as parameters.
    window = [re.sub(r"#.*$", "", line) for line in lines[index:index + 8]]
    chunk = "\n".join(window)
    start = chunk.find("(")
    if start < 0:
        return "", index + 1
    depth = 0
    for offset in range(start, len(chunk)):
        char = chunk[offset]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                args = chunk[start + 1:offset]
                consumed = chunk[:offset].count("\n")
                return re.sub(r"\s+", " ", args).strip(), index + consumed + 1
    return "", index + 1


def _hint(lines: list[str], body: int) -> str:
    """First docstring line, else the first real body line."""
    for line in lines[body:body + 3]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = DOC_RE.match(line)
        if match is not None:
            quote = match.group(1)
            content = stripped
            for prefix in ('"""', "'''", '"', "'"):
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break
            if quote in ('"""', "'''") and content.endswith(quote):
                content = content[:-3]
            elif content.endswith("\\") is False and len(quote) == 1 and content.endswith(quote):
                content = content[:-1]
            first = content.strip().split("\\n")[0].strip()
            if first:
                return first[:60]
            continue
        return stripped[:60]
    return ""


def extract_candidates(rel: str, text: str) -> list[dict]:
    """All function candidates in one file, plus its module-level candidate."""
    lines = text.splitlines()
    candidates: list[dict] = []
    # Enclosing scopes (classes and functions), outermost first. A nested
    # definition is qualified by every scope still open at its indentation.
    scopes: list[tuple[int, str]] = []
    seen: Counter = Counter()
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        width = indent_of(line)
        class_match = CLASS_RE.match(line)
        if class_match is not None:
            while scopes and scopes[-1][0] >= width:
                scopes.pop()
            scopes.append((width, class_match.group(2)))
            continue
        def_match = DEF_RE.match(line)
        if def_match is None:
            while scopes and width <= scopes[-1][0]:
                scopes.pop()
            continue
        while scopes and scopes[-1][0] >= width:
            scopes.pop()
        name = def_match.group(2)
        qualname = ".".join([entry[1] for entry in scopes] + [name])
        args, body = _signature(lines, index)
        display_args = ", ".join(
            part.strip() for part in args.split(",") if part.strip() not in ("self", "cls")
        ).strip()
        end = len(lines)
        for probe in range(index + 1, len(lines)):
            candidate_line = lines[probe]
            if not candidate_line.strip() or candidate_line.lstrip().startswith("#"):
                continue
            probe_width = indent_of(candidate_line)
            if DEF_RE.match(candidate_line) is not None or CLASS_RE.match(candidate_line) is not None:
                if probe_width <= width:
                    end = probe
                    break
            elif probe_width < width:
                end = probe
                break
        hint = _hint(lines, body)
        base = f"{rel}:{qualname}"
        seen[base] += 1
        ident = base if seen[base] == 1 else f"{base}#{index + 1}"
        candidate_text = f"{rel}:{qualname}({display_args})"
        if hint:
            candidate_text = f"{candidate_text} \u2014 {hint}"
        scopes.append((width, name))
        candidates.append({
            "id": ident,
            "path": rel,
            "qualname": qualname,
            "kind": "function",
            "line": index + 1,
            "end": end,
            "args": args,
            "text": candidate_text[:MAX_TEXT],
        })
    candidates.append({
        "id": f"{rel}::<module>",
        "path": rel,
        "qualname": "<module>",
        "kind": "module",
        "line": 1,
        "end": len(lines),
        "args": "",
        "text": f"{rel} (module level)"[:MAX_TEXT],
    })
    return candidates


# --- ranking ----------------------------------------------------------------

def bm25(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 scores for every document against one query."""
    if not docs:
        return []
    n = len(docs)
    lengths = [len(doc) for doc in docs]
    average = sum(lengths) / n if n else 0.0
    frequencies = []
    for doc in docs:
        frequencies.append(Counter(doc))
    document_frequency = Counter()
    for counts in frequencies:
        for term in counts:
            document_frequency[term] += 1
    scores = []
    for counts, length in zip(frequencies, lengths):
        score = 0.0
        for term in query:
            if counts[term] == 0:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            norm = counts[term] * (k1 + 1) / (
                counts[term] + k1 * (1 - b + b * (length / average if average else 0.0))
            )
            score += idf * norm
        scores.append(score)
    return scores


def _path_match(rel: str, query: set[str]) -> float:
    stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    pieces = set(re.split(r"[_\-.]", stem.lower()))
    return 1.0 if pieces & query else 0.0


def rank_candidates(
    candidates: list[dict],
    failure: str,
    task: str,
    k: int = 12,
    root: str = "",
) -> dict:
    """Weighted shortlist: frame evidence first, then BM25 over text."""
    if not candidates:
        return {"ranked": [], "stratum": "none", "frames": []}
    failure_text = query_text(failure)
    frames = parse_frames(failure_text, root)
    source_frames = [frame for frame in frames if not is_test_path(frame["file"])]
    frame_files = {frame["file"] for frame in source_frames}
    query = set(tokens(failure_text)) | set(tokens(task))
    failure_tokens = tokens(failure_text)
    task_tokens = tokens(task)

    texts = [candidate["text"] for candidate in candidates]
    docs = [tokens(text) for text in texts]
    failure_scores = bm25(failure_tokens, docs)
    task_scores = bm25(task_tokens, docs)
    # Normalise each BM25 signal to [0, 1] so the weights mean what they say.
    # Raw BM25 grows with corpus size and match count, which would otherwise let
    # one signal swamp the frame evidence on a large repo.
    failure_peak = max(failure_scores) if failure_scores else 0.0
    task_peak = max(task_scores) if task_scores else 0.0

    scored = []
    for position, candidate in enumerate(candidates):
        frame_hit = 0.0
        if candidate["kind"] == "function":
            for frame in source_frames:
                if frame["file"] == candidate["path"] and candidate["line"] <= frame["line"] <= candidate["end"]:
                    frame_hit = 1.0
                    break
        file_hit = 1.0 if candidate["path"] in frame_files else 0.0
        score = (
            6.0 * frame_hit
            + 3.0 * file_hit
            + 2.0 * _path_match(candidate["path"], query)
            + 1.5 * (failure_scores[position] / failure_peak if failure_peak else 0.0)
            + 1.0 * (task_scores[position] / task_peak if task_peak else 0.0)
        )
        scored.append((score, candidate))
    scored.sort(key=lambda row: (-row[0], len(row[1]["path"]), row[1]["line"], row[1]["id"]))
    ranked = [candidate for _, candidate in scored[:k]]
    if source_frames:
        stratum = "frame"
    elif frames:
        stratum = "assert-only"
    else:
        stratum = "none"
    return {"ranked": ranked, "stratum": stratum, "frames": frames}


def shortlist_recall(candidates: list[dict], gold: str, k: int, failure: str, task: str, root: str = "") -> float:
    ranked = rank_candidates(candidates, failure, task, k, root)["ranked"]
    return 1.0 if any(candidate["id"] == gold for candidate in ranked) else 0.0


# --- wire body --------------------------------------------------------------

def build_choice(candidates: list[dict], task: str, failure: str, tail: int = 2000) -> dict:
    """The exact `/v1/systemone` body the probe and the Host must both send."""
    # Keep the head: the focused text starts with the failing node ids and the
    # first traceback, which is exactly what must survive a budget cut.
    output = (failure or "")[:tail] if tail > 0 else (failure or "")
    return {
        "state": {
            "task": (task or "")[:MAX_TASK],
            "changes": [{"tool": "pytest", "output": output}],
        },
        "questions": {
            "loc": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {candidate["id"]: candidate["text"] for candidate in candidates},
            },
        },
    }


def rank_from_probabilities(candidates: list[dict], probabilities: dict) -> list[tuple[str, float]]:
    """Ordered (candidate id, probability) pairs, highest first."""
    rows = [
        (candidate["id"], float(probabilities.get(candidate["id"], 0.0)))
        for candidate in candidates
    ]
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows


# --- CLI --------------------------------------------------------------------

def _self_check() -> int:
    sample = (
        "import os\n"
        "\n"
        "class Spooled(IOBase):\n"
        "    \"\"\"A spooled buffer.\"\"\"\n"
        "\n"
        "    def tell(self) -> int:\n"
        "        \"\"\"Return the codepoint position\"\"\"\n"
        "        return self._tell + 1\n"
        "\n"
        "    @property\n"
        "    def length(self):\n"
        "        return self._tell\n"
        "\n"
        "\n"
        "def helper(value, other=1) -> int:\n"
        "    return value + other\n"
    )
    found = extract_candidates("pkg/ioutils.py", sample)
    ids = [candidate["id"] for candidate in found]
    assert "pkg/ioutils.py:Spooled.tell" in ids, ids
    assert "pkg/ioutils.py:helper" in ids, ids
    assert "pkg/ioutils.py::<module>" in ids, ids
    tell = next(candidate for candidate in found if candidate["id"] == "pkg/ioutils.py:Spooled.tell")
    assert tell["line"] == 6 and tell["end"] == 10, tell
    assert "codepoint position" in tell["text"], tell["text"]
    helper = next(candidate for candidate in found if candidate["id"] == "pkg/ioutils.py:helper")
    assert helper["args"] == "value, other=1", helper["args"]
    assert "value + other" in helper["text"], helper["text"]

    failure = (
        "    def test_tell(self):\n"
        ">       assert buf.tell() == 12\n"
        "E       assert 13 == 12\n"
        "\n"
        "pkg/ioutils.py:6: in tell\n"
        "    return self._tell + 1\n"
        "E   AssertionError: bad position\n"
        "FAILED tests/test_ioutils.py::TestSpooled::test_tell - AssertionError: bad position\n"
    )
    frames = parse_frames(failure, "/repo")
    assert frames == [{"file": "pkg/ioutils.py", "line": 6, "func": "tell"}], frames
    assert parse_failed_tests(failure) == ["tests/test_ioutils.py::TestSpooled::test_tell"]

    ranked = rank_candidates(found, failure, "buffer position is off by one", 3)
    assert ranked["stratum"] == "frame"
    assert ranked["ranked"][0]["id"] == "pkg/ioutils.py:Spooled.tell", ranked["ranked"][0]
    body = build_choice(ranked["ranked"], "task", failure, tail=500)
    assert body["questions"]["loc"]["criteria"][ranked["ranked"][0]["id"]] == "pkg/ioutils.py:Spooled.tell() — Return the codepoint position"
    assert body["state"]["task"] == "task"
    assert is_test_path("tests/test_x.py") and is_test_path("pkg/test_y.py")
    assert not is_test_path("pkg/testing_utils.py")
    assert strip_root('File "/testbed/pkg/x.py", line 1, in f', "/data/repo") == 'File "pkg/x.py", line 1, in f'
    print("self-check ok: %d candidates" % len(found))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", metavar="REPO", help="print the frozen candidate list for a repo")
    parser.add_argument("--out", help="write the dump here instead of stdout")
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return _self_check()
    if not args.dump:
        parser.error("pass --dump or --self-check")
    candidates = []
    for item in walk_repo(args.dump, max_files=args.max_files):
        candidates.extend(extract_candidates(item["path"], item["text"]))
    payload = {"repo": os.path.abspath(args.dump), "count": len(candidates), "candidates": candidates}
    text = json.dumps(payload, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print(f"{len(candidates)} candidates -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
