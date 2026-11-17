"""Hierarchical narrowing: file -> section -> function -> line window.

The single-shot endpoint ranks hunks of one diff, and the flat probe ranks a
BM25+frame shortlist. This is the third shape: a cascade that asks Laya a small
choice question at each level over a *real* hierarchy (directory -> file ->
class -> function -> ~10-line window), with a beam so one bad early answer does
not end the search.

    python scripts/localize_narrow.py --root /data/laya-ft/failure-loc --beam 3

Scoring: every node inherits `score = parent_score * P(node | parent)`, so the
leaves can be ranked directly against the flat baselines and the gold function
ids from the corpus.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import localize_candidates as lc  # noqa: E402
import localize_failure as lf  # noqa: E402

MAX_OPTIONS = 48
INSTRUCTIONS = "A test is failing. Which of these is most likely to contain the bug?"
WINDOW = 10


# --- tree -------------------------------------------------------------------

def load_files(item: dict) -> dict[str, list[dict]]:
    """repo-relative path -> frozen candidate list for that file."""
    files: dict[str, list[dict]] = {}
    for blob in lc.walk_repo(item["repo"]):
        files[blob["path"]] = lc.extract_candidates(blob["path"], blob["text"])
    return files


def file_node(path: str, candidates: list[dict]) -> dict:
    """A file's children: its classes, its module-level functions, its module leaf."""
    by_class: dict[str, list[dict]] = {}
    module_level: list[dict] = []
    for candidate in candidates:
        if candidate["kind"] != "function":
            continue
        if "." in candidate["qualname"]:
            owner = candidate["qualname"].rsplit(".", 1)[0]
            by_class.setdefault(owner, []).append(candidate)
        else:
            module_level.append(candidate)
    children: list[dict] = []
    for owner, members in sorted(by_class.items()):
        children.append({
            "kind": "class",
            "id": f"{path}:class:{owner}",
            "label": owner.rsplit(".", 1)[-1],
            "text": f"{owner.rsplit('.', 1)[-1]} class, {len(members)} methods",
            "children": [function_leaf(candidate) for candidate in members],
            "order": min(member["line"] for member in members),
        })
    children.extend(function_leaf(candidate) for candidate in module_level)
    module = next((candidate for candidate in candidates if candidate["kind"] == "module"), None)
    if module is not None:
        children.append({
            "kind": "leaf", "id": module["id"], "label": "<module>",
            "text": "module-level code", "order": 0, "candidate": module,
        })
    children.sort(key=lambda node: node.get("order", 0))
    return {
        "kind": "file",
        "id": f"file:{path}",
        "label": os.path.basename(path),
        "text": f"{os.path.basename(path)}, {max(0, len(candidates) - 1)} definitions",
        "children": children,
    }


def function_leaf(candidate: dict) -> dict:
    text = candidate["text"]
    if ": " in text:
        text = text.split(": ", 1)[1]
    return {
        "kind": "leaf",
        "id": candidate["id"],
        "label": candidate["qualname"].rsplit(".", 1)[-1],
        "text": text[:150],
        "order": candidate["line"],
        "candidate": candidate,
    }


def dir_node(prefix: str, files: dict[str, list[dict]]) -> dict:
    """A directory's children: immediate subdirectories and files."""
    depth = len([part for part in prefix.split("/") if part]) if prefix else 0
    subdirs: set[str] = set()
    direct: list[str] = []
    for path in files:
        parts = path.split("/")
        if prefix and not path.startswith(prefix + "/"):
            continue
        rest = parts[depth:]
        if len(rest) == 1:
            direct.append(path)
        else:
            subdirs.add(rest[0])
    children: list[dict] = []
    for name in sorted(subdirs):
        child_prefix = f"{prefix}/{name}" if prefix else name
        owned = [path for path in files if path.startswith(child_prefix + "/")]
        symbols: list[str] = []
        for path in owned[:2]:
            symbols.extend(candidate["qualname"].rsplit(".", 1)[-1] for candidate in files[path][:2])
        children.append({
            "kind": "dir",
            "id": f"dir:{child_prefix}",
            "label": name,
            "text": f"{name}/ ({len(owned)} files){': ' + ', '.join(symbols[:3]) if symbols else ''}",
            "prefix": child_prefix,
            "order": 0,
        })
    for path in sorted(direct):
        children.append(file_node(path, files[path]))
    return {
        "kind": "dir",
        "id": f"dir:{prefix}",
        "label": prefix.rsplit("/", 1)[-1] or "(root)",
        "text": f"{prefix or '(root)'}/",
        "prefix": prefix,
        "children": children,
    }


def cap(children: list[dict]) -> list[dict]:
    """Split a wide level into contiguous 'part k of N' sections."""
    if len(children) <= MAX_OPTIONS:
        return children
    size = math.ceil(len(children) / MAX_OPTIONS)
    buckets = []
    for index in range(0, len(children), size):
        chunk = children[index:index + size]
        names = ", ".join(node["label"] for node in chunk[:3])
        buckets.append({
            "kind": "bucket",
            "id": f"bucket:{children[0]['id']}:{index // size}",
            "label": f"part {index // size + 1}",
            "text": f"part {index // size + 1} of {math.ceil(len(children) / size)}: {names}",
            "children": chunk,
            "order": index,
        })
    return buckets


# --- querying ---------------------------------------------------------------

def ask(base_url: str, state: dict, criteria: dict) -> dict[str, float]:
    body = {"state": state, "questions": {"loc": {
        "type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}}
    request = urllib.request.Request(
        f"{base_url}/v1/systemone", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())["answers"]["loc"].get("probabilities") or {}


def state_for(item: dict, trail: list[str], failure_chars: int = 1200) -> dict:
    task = item["problem"][:300]
    if trail:
        task += "\nNarrowing so far: " + " > ".join(trail[-4:])
    return {
        "task": task[:400],
        "changes": [{"tool": "pytest", "output": lc.focus_failure(item["failure"], failure_chars)}],
    }


def option_text(parent: dict, option: dict) -> str:
    """Descriptions help for containers; bare names survive the option budget."""
    if parent["kind"] in ("file", "class") and option["kind"] in ("class", "leaf"):
        return option["label"][:48]
    if parent["kind"] == "bucket" and option["kind"] in ("class", "leaf"):
        return option["label"][:48]
    return option["text"][:150]


def expand(base_url: str, item: dict, node: dict, trail: list[str], files: dict) -> list[tuple[dict, float]]:
    """Ask one question about a node's children; return (child, probability)."""
    kids = node.get("children")
    if not kids:
        return []
    options = cap(kids)
    criteria = {
        f"n{index + 1}": option_text(node, option)
        for index, option in enumerate(options)
    }
    probabilities = ask(base_url, state_for(item, trail), criteria)
    scored = []
    for index, option in enumerate(options):
        probability = float(probabilities.get(f"n{index + 1}", 0.0))
        scored.append((option, probability))
    scored.sort(key=lambda row: -row[1])
    return scored


def narrow(item: dict, base_url: str, beam: int, depth: int = 8) -> tuple[list[str], list[dict], int]:
    files = load_files(item)
    root = dir_node("", files)
    frontier = [(root, 1.0, [])]
    leaves: list[tuple[str, float]] = []
    calls = 0
    stages: list[dict] = []
    for _ in range(depth):
        expanded: list[tuple[dict, float, list[str]]] = []
        for node, score, trail in frontier:
            node = dir_node(node["prefix"], files) if node["kind"] == "dir" and "children" not in node else node
            if node["kind"] == "leaf":
                leaves.append((node["id"], score))
                continue
            kids = node.get("children") or []
            if not kids:
                continue
            if len(kids) == 1 and node["kind"] not in ("leaf",):
                # Unambiguous level: descend without spending a call.
                expanded.append((kids[0], score, trail + [kids[0]["label"]]))
                continue
            scored = expand(base_url, item, node, trail, files)
            calls += 1
            stages.append({
                "node": node["id"],
                "options": len(scored),
                "top": [(kid["id"], round(probability, 3)) for kid, probability in scored[:3]],
            })
            for kid, probability in scored[:max(1, beam)]:
                if probability <= 0.0:
                    continue
                expanded.append((kid, score * probability, trail + [kid["label"]]))
        if not expanded:
            break
        expanded.sort(key=lambda row: -row[1])
        frontier = expanded[:beam]
        if all(node["kind"] == "leaf" for node, _, _ in frontier):
            leaves.extend((node["id"], score) for node, score, _ in frontier)
            break
    leaves.sort(key=lambda row: -row[1])
    ordered = [ident for ident, _ in leaves]
    return ordered, stages, calls


def narrow_files(item: dict, base_url: str, beam: int, depth: int = 8) -> tuple[list[tuple[str, float]], list[dict], int]:
    """Cascade only until files: the level where semantics can beat BM25."""
    files = load_files(item)
    frontier = [(dir_node("", files), 1.0, [])]
    selected: list[tuple[str, float]] = []
    stages: list[dict] = []
    calls = 0
    for _ in range(depth):
        expanded: list[tuple[dict, float, list[str]]] = []
        for node, score, trail in frontier:
            node = dir_node(node["prefix"], files) if node["kind"] == "dir" and "children" not in node else node
            if node["kind"] == "file":
                selected.append((node["id"].split(":", 1)[1], score))
                continue
            kids = node.get("children") or []
            if not kids:
                continue
            if len(kids) == 1:
                expanded.append((kids[0], score, trail + [kids[0]["label"]]))
                continue
            scored = expand(base_url, item, node, trail, files)
            calls += 1
            stages.append({"node": node["id"], "options": len(scored),
                           "top": [(kid["id"], round(probability, 3)) for kid, probability in scored[:3]]})
            for kid, probability in scored[:max(1, beam)]:
                if probability > 0.0:
                    expanded.append((kid, score * probability, trail + [kid["label"]]))
        if not expanded:
            break
        expanded.sort(key=lambda row: -row[1])
        frontier = expanded[:beam]
        if all(node["kind"] == "file" for node, _, _ in frontier):
            selected.extend((node["id"].split(":", 1)[1], score) for node, score, _ in frontier)
            break
    selected.sort(key=lambda row: -row[1])
    return selected, stages, calls


def hybrid(item: dict, base_url: str, beam: int, per_file: int = 8, tail: int = 1200) -> tuple[list[str], dict]:
    """Cascade to a few files, then rank functions inside each chosen file."""
    chosen, stages, calls = narrow_files(item, base_url, beam)
    pool: list[tuple[float, str]] = []
    file_hits = []
    for path, probability in chosen[:3]:
        members = [candidate for candidate in item["candidates"] if candidate["path"] == path]
        if path in item["gold"]["files"]:
            file_hits.append(path)
        if not members:
            continue
        ranked = lc.rank_candidates(members, item["failure"], item["problem"], per_file, item["repo"])
        shortlist = ranked["ranked"]
        criteria = {f"h{index + 1}": candidate["text"].split(": ", 1)[-1][:140]
                    for index, candidate in enumerate(shortlist)}
        state = {
            "task": item["problem"][:400],
            "changes": [{"tool": "pytest", "output": lc.focus_failure(item["failure"], tail)}],
        }
        probabilities = ask(base_url, state, criteria)
        calls += 1
        for index, candidate in enumerate(shortlist):
            pool.append((probability * float(probabilities.get(f"h{index + 1}", 0.0)), candidate["id"]))
    pool.sort(key=lambda row: -row[0])
    ordered = [ident for _, ident in pool]
    flat = lc.rank_candidates(item["candidates"], item["failure"], item["problem"], 12, item["repo"])
    seen = set(ordered)
    ordered.extend(candidate["id"] for candidate in flat["ranked"] if candidate["id"] not in seen)
    return ordered, {"calls": calls, "files": [path for path, _ in chosen[:3]],
                     "gold_file_hit": bool(file_hits), "stages": stages}


def window_eval(item: dict, base_url: str, beams: int = 3) -> dict:
    """Within the gold function, rank 10-line windows; isolates granularity."""
    candidates = {candidate["id"]: candidate for candidate in item["candidates"]}
    results = []
    for site in item["gold"]["sites"]:
        candidate = candidates.get(site)
        if candidate is None or candidate["kind"] != "function":
            continue
        gold_lines = [
            line for line in item["gold"]["files"].get(candidate["path"], [])
            if candidate["line"] <= line <= candidate["end"]
        ]
        if not gold_lines:
            continue
        windows = []
        for start in range(candidate["line"], candidate["end"] + 1, WINDOW):
            end = min(start + WINDOW - 1, candidate["end"])
            windows.append({"start": start, "end": end})
        if len(windows) < 2:
            continue
        try:
            lines = open(os.path.join(item["repo"], candidate["path"]), errors="replace").read().splitlines()
            body = "\n".join(lines[candidate["line"] - 1:candidate["end"]])
        except OSError:
            continue
        criteria = {
            f"w{index + 1}": f"lines {window['start']}-{window['end']}"
            for index, window in enumerate(windows[:MAX_OPTIONS * 3])
        }
        state = {
            "task": f"{item['problem'][:200]}\nFunction under review:\n{body[:500]}",
            "changes": [{"tool": "pytest", "output": lc.focus_failure(item["failure"], 600)}],
        }
        probabilities = ask(base_url, state, criteria)
        ordered = sorted(probabilities.items(), key=lambda row: -row[1])
        gold_window = next(
            (index for index, window in enumerate(windows)
             if any(window["start"] <= line <= window["end"] for line in gold_lines)),
            None,
        )
        if gold_window is None:
            continue
        label = f"w{gold_window + 1}"
        rank = next((index for index, (key, _) in enumerate(ordered, 1) if key == label), len(windows) + 1)
        results.append({"sites": [rank], "first": rank, "windows": len(windows)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/laya-ft/failure-loc")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--lines", action="store_true", help="also rank windows inside the gold function")
    parser.add_argument("--hybrid", action="store_true", help="cascade to files, then rank functions within them")
    parser.add_argument("--only", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    instances = lf.discover(args.root)
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        instances = [item for item in instances if item["instance_id"] in wanted]
    if args.limit:
        instances = instances[:args.limit]
    if not instances:
        print(f"no usable instances under {args.root}", flush=True)
        return 1
    print(f"instances={len(instances)} beam={args.beam}", flush=True)

    rows = []
    started = time.time()
    for index, item in enumerate(instances, 1):
        began = time.time()
        hybrid_info = None
        if args.hybrid:
            ordered, hybrid_info = hybrid(item, args.base_url, args.beam)
            stages, calls = hybrid_info["stages"], hybrid_info["calls"]
        else:
            ordered, stages, calls = narrow(item, args.base_url, args.beam)
        flat = lc.rank_candidates(item["candidates"], item["failure"], item["problem"], 12, item["repo"])
        fallback = [candidate["id"] for candidate in flat["ranked"]]
        ordered = ordered + [ident for ident in fallback if ident not in set(ordered)]
        row = {
            "instance_id": item["instance_id"],
            "stratum": flat["stratum"],
            "sites": item["gold"]["sites"],
            "calls": calls,
            "ms": round((time.time() - began) * 1000, 1),
            "narrow": {"ranks": lf.ranks(ordered, item["gold"]["sites"]), "ordered": ordered[:20]},
            "stages": stages,
        }
        if hybrid_info is not None:
            row["hybrid"] = {"files": hybrid_info["files"], "gold_file_hit": hybrid_info["gold_file_hit"]}
        if args.lines:
            windows = window_eval(item, args.base_url)
            if windows:
                row["windows"] = {"ranks": {"sites": [w["first"] for w in windows], "first": min(w["first"] for w in windows)}, "n": len(windows)}
        rows.append(row)
        print(f"[{index}/{len(instances)}] {item['instance_id'][:44]:44s} calls={calls:2d} "
              f"rank={row['narrow']['ranks']['first']}", flush=True)

    flat_rows = [{"instance_id": row["instance_id"], "stratum": row["stratum"],
                  "sites": row["sites"], "shortlist": row["narrow"]["ordered"][:12],
                  "narrow": row["narrow"]["ranks"]} for row in rows]
    summary = {"narrow": lf.summarize(flat_rows, "narrow")}
    # Deterministic control on the same corpus.
    control = []
    for item, row in zip(instances, rows):
        r = lc.rank_candidates(item["candidates"], item["failure"], item["problem"], 12, item["repo"])
        control.append({"instance_id": item["instance_id"], "stratum": row["stratum"],
                        "sites": item["gold"]["sites"],
                        "shortlist": [candidate["id"] for candidate in r["ranked"]],
                        "H3": lf.ranks([candidate["id"] for candidate in r["ranked"]], item["gold"]["sites"])})
    summary["H3"] = lf.summarize(control, "H3")
    payload = {
        "root": os.path.abspath(args.root),
        "beam": args.beam,
        "instances": len(rows),
        "calls": sum(row["calls"] for row in rows),
        "server_seconds": round(time.time() - started, 1),
        "summary": summary,
        "lines": {
            "functions_evaluated": len([row for row in rows if "windows" in row]),
            "windows": sum(row.get("windows", {}).get("n", 0) for row in rows),
            "site_top1": round(
                sum(1 for row in rows if row.get("windows", {}).get("ranks", {}).get("first") == 1)
                / max(1, len([row for row in rows if "windows" in row])), 3),
        },
        "rows": rows,
    }
    print(json.dumps({"summary": summary, "lines": payload["lines"],
                      "calls": payload["calls"]}, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
