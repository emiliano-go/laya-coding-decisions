"""Decision dataset v6: execution-verified sources + bug_class + one layout.

What this adds over v2/m5f:
  - canonical state layout shared with dsh-totem (`scripts/normalize_state.py`);
  - execution/dataset-verified positives from SWE-bench-extra and R2E-Gym-Lite;
  - paired correct/incorrect responses from CodeJudgeBench (codegen + coderepair)
    for requirements_met and bug_class;
  - AXIOM-style perturbations of verified patches with the mutated bug class;
  - adversarial CodeJudgeBench cases written to a separate OOD file, never trained.

Base rows are the already-corrected m5f splits. New rows are repeated
(default 2x) because they are far fewer than the 255k base rows.

Usage:
  python scripts/gen_dataset_v6.py --offline          # local cache + base only
  python scripts/gen_dataset_v6.py                    # also stream HF sources
  python scripts/gen_dataset_v6.py --limit 2000       # smoke test
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_dataset_v2 import derive_memories, noul, score  # noqa: E402
from normalize_state import normalize_state, task_split  # noqa: E402

random.seed(23)
BASE = os.environ.get("LAYACD_DATA", "/data/laya-ft")
MAX_CHARS = 4000

BUG_CLASS = {
    "none": "No bug is likely",
    "off-by-one": "Off-by-one, boundary, or loop/inclusive-range error",
    "null-empty": "Missing null/empty/undefined handling",
    "wrong-comparison": "Wrong comparison or operator (equality, ordering, arithmetic)",
    "scope-unrelated": "Unrequested or unrelated changes / scope creep",
    "swallowed-exception": "Exception swallowed or errors hidden",
    "resource-leak": "Unclosed resource or cleanup missing",
    "type-coercion": "Wrong type, coercion, or serialization issue",
    "other": "A different bug class",
}

QUESTIONS = {
    "requirements_met": {
        "type": "noul",
        "instructions": "The code correctly implements every requirement in the task, including all edge cases.",
    },
    "bug_risk": {
        "type": "score",
        "instructions": "Does the code contain a bug or miss an edge case?",
        "criteria": ["No bug, the code is correct", "Minor risk", "Likely a bug", "Definitely buggy"],
    },
    "bug_class": {
        "type": "choice",
        "instructions": "If the change likely introduces a bug, what kind of bug is it? Choose the best match, or 'none' if no bug is likely.",
        "criteria": BUG_CLASS,
    },
}


def choice_gold(label: str) -> dict:
    keys = list(BUG_CLASS)
    other = round(0.1 / (len(keys) - 1), 4)
    return {
        "probabilities": {key: (0.9 if key == label else other) for key in keys},
        "label": keys.index(label),
    }


def clean_task(task: str) -> str:
    """swe_smith_traj stores a stringified list of content parts; unwrap it."""
    task = (task or "").strip()
    if task.startswith("[{") and "'type'" in task:
        try:
            parts = ast.literal_eval(task)
            task = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except (ValueError, SyntaxError):
            pass
    return task.strip()


def make_row(
    task: str,
    diff: str,
    req: bool,
    bug: int,
    bug_class: str | None,
    *,
    answer: str = "applied the change",
    source: str = "",
    validated: str = "synthetic",
    path: str = "",
) -> dict | None:
    task = clean_task(task)
    diff = (diff or "").strip()[:MAX_CHARS]
    if not task or not diff:
        return None
    state = normalize_state(
        task=task,
        answer=answer,
        changes=[{"tool": "apply_patch", "input": diff, "output": "applied", **({"path": path} if path else {})}],
        tool_calls=[{"tool": "apply_patch", "title": "patch", "status": "completed"}],
        grounding=[],
        memory=derive_memories(task),
    )
    gold = {"requirements_met": noul(req), "bug_risk": score(bug)}
    if bug_class is not None:
        gold["bug_class"] = choice_gold(bug_class)
    return {
        "state": state,
        "questions": json.dumps(QUESTIONS),
        "gold": json.dumps(gold),
        "split": task_split(task),
        "source": source,
        "validated": validated,
    }


def normalize_row(row: dict) -> dict | None:
    state = row.get("state")
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            return None
    if not isinstance(state, dict):
        return None
    task = clean_task(state.get("task", ""))
    changes = state.get("changes") if isinstance(state.get("changes"), list) else []
    row["state"] = normalize_state(
        task=task,
        answer=state.get("answer", ""),
        changes=changes,
        tool_calls=state.get("toolCalls") if isinstance(state.get("toolCalls"), list) else [],
        grounding=state.get("grounding") if isinstance(state.get("grounding"), list) else [],
        memory=state.get("memory") if isinstance(state.get("memory"), list) else [],
    )
    questions = row.get("questions")
    gold = row.get("gold")
    row["questions"] = questions if isinstance(questions, str) else json.dumps(questions)
    row["gold"] = gold if isinstance(gold, str) else json.dumps(gold)
    return row


def load_base(split: str, limit: int) -> list[dict]:
    path = os.path.join(BASE, f"{split}_m5f.jsonl")
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        row = normalize_row(json.loads(line))
        if row is not None:
            rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def as_diff(code: str, path: str = "solution.py") -> str:
    lines = (code or "").strip().splitlines()
    if not lines:
        return ""
    body = "\n".join("+" + line for line in lines)
    return f"--- a/{path}\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"


def classify(text: str) -> str:
    value = (text or "").lower()
    if "off-by" in value or "index" in value or "out of range" in value or "boundary" in value:
        return "off-by-one"
    if "none" in value or "null" in value or "attributeerror" in value or "undefined" in value:
        return "null-empty"
    if "type" in value or "coerc" in value or "serial" in value:
        return "type-coercion"
    if "compar" in value or "operator" in value or "!=" in value or "==" in value:
        return "wrong-comparison"
    if "exception" in value or "swallow" in value or "catch" in value:
        return "swallowed-exception"
    return "other"


def stream(dataset: str, split: str, config: str | None = None):
    from datasets import load_dataset

    return load_dataset(dataset, config, split=split, streaming=True)


def from_swe_bench_extra(limit: int) -> list[dict]:
    rows = []
    for item in stream("nebius/SWE-bench-extra", "train"):
        row = make_row(
            item.get("problem_statement", ""),
            item.get("patch", ""),
            True,
            0,
            "none",
            source="swe-bench-extra",
            validated="dataset-test",
        )
        if row is not None:
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def r2e_diff(item: dict) -> str:
    try:
        parsed = json.loads(item.get("parsed_commit_content") or "{}")
    except (TypeError, ValueError):
        return ""
    chunks = []
    for file in parsed.get("file_diffs", []):
        old = file.get("old_file_content")
        new = file.get("new_file_content")
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        header = file.get("minus_file") or {}
        if isinstance(header, str):
            try:
                header = ast.literal_eval(header)
            except (ValueError, SyntaxError):
                header = {}
        path = ""
        if isinstance(header, dict):
            path = (header.get("path") or "").removeprefix("a/")
        if not path:
            header = file.get("header") or {}
            if isinstance(header, str):
                try:
                    header = ast.literal_eval(header)
                except (ValueError, SyntaxError):
                    header = {}
            if isinstance(header, dict):
                path = ((header.get("file") or {}).get("path") or "")
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
            )
        )
        if diff:
            chunks.append(diff)
    return "\n".join(chunks)


def r2e_ok(item: dict) -> bool:
    try:
        expected = json.loads(item.get("expected_output_json") or "{}")
    except (TypeError, ValueError):
        return False
    if not isinstance(expected, dict) or not expected:
        return False
    return all(str(value).upper() == "PASSED" for value in expected.values())


def from_r2e(limit: int) -> list[dict]:
    rows = []
    for item in stream("R2E-Gym/R2E-Gym-Lite", "train"):
        diff = r2e_diff(item)
        if not diff:
            continue
        task = item.get("problem_statement", "")
        if r2e_ok(item):
            row = make_row(task, diff, True, 0, "none", source="r2e-gym", validated="execution")
        else:
            row = make_row(task, diff, False, 3, "other", source="r2e-gym", validated="execution")
        if row is not None:
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def from_codejudge(limit: int) -> list[dict]:
    rows = []
    for model in ("gemini_2.5_pro", "claude_4_sonnet", "gemini_2.5_flash"):
        try:
            items = stream("mattymchen/codejudgebench", model, config="codegen")
        except Exception as error:
            print(f"  codejudge codegen/{model} unavailable: {error}", flush=True)
            continue
        for item in items:
            task = f"{item.get('question_title', '')}\n\n{item.get('question_content', '')}"
            for key, req, bug in (("pos_response", True, 0), ("neg_response", False, 3)):
                code = item.get(key) or ""
                row = make_row(
                    task, as_diff(code), req, bug,
                    "none" if req else "other",
                    source="codejudge-codegen", validated="dataset-execution",
                )
                if row is not None:
                    rows.append(row)
            if len(rows) >= limit:
                return rows
    for model in ("gemini_2.5_pro", "claude_3.7_sonnet"):
        try:
            items = stream("mattymchen/codejudgebench", model, config="coderepair")
        except Exception as error:
            print(f"  codejudge coderepair/{model} unavailable: {error}", flush=True)
            continue
        for item in items:
            task = f"{item.get('question_title', '')}\n\n{item.get('question_content', '')}"
            meta = item.get("wrong_meta") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            label = classify(" ".join(str(meta.get(key, "")) for key in ("error", "error_message")))
            row = make_row(
                task, as_diff(item.get("wrong_code") or ""), False, 3, label,
                source="codejudge-coderepair", validated="dataset-execution",
            )
            if row is not None:
                rows.append(row)
            row = make_row(
                task, as_diff(item.get("pos_response") or ""), True, 0, "none",
                source="codejudge-coderepair", validated="dataset-execution",
            )
            if row is not None:
                rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


MUTATIONS = [
    ("==", "=", "wrong-comparison"),
    ("<=", "<", "off-by-one"),
    (">=", ">", "off-by-one"),
    ("!=", "==", "wrong-comparison"),
    ("+ 1", "+ 2", "off-by-one"),
    ("- 1", "- 2", "off-by-one"),
    ("return True", "return False", "wrong-comparison"),
    ("return False", "return True", "wrong-comparison"),
    ("min(", "max(", "wrong-comparison"),
    ("max(", "min(", "wrong-comparison"),
    ("sorted(", "list(", "other"),
    (" and ", " or ", "wrong-comparison"),
    (" + ", " - ", "wrong-comparison"),
]


def perturb(diff: str) -> tuple[str, str] | None:
    lines = diff.split("\n")
    added = [index for index, line in enumerate(lines) if line.startswith("+") and not line.startswith("+++")]
    random.shuffle(added)
    for index in added:
        for old, new, label in MUTATIONS:
            if old in lines[index]:
                lines[index] = lines[index].replace(old, new, 1)
                return "\n".join(lines), label
    return None


def perturb_rows(rows: list[dict], limit: int) -> list[dict]:
    out = []
    for row in rows:
        if len(out) >= limit:
            break
        state = row["state"]
        diff = state["changes"][0].get("diff") or state["changes"][0].get("input", "")
        mutated = perturb(diff)
        if mutated is None:
            continue
        diff_text, label = mutated
        new = make_row(
            state["task"], diff_text, True, 3, label,
            source=f"{row.get('source', 'base')}-perturbed", validated="synthetic",
        )
        if new is not None:
            out.append(new)
    return out


def cross_rows(rows: list[dict], limit: int) -> list[dict]:
    out = []
    for index in range(min(limit, max(0, len(rows) - 1))):
        first, second = rows[index], rows[(index + 1 + random.randrange(len(rows) - 1)) % len(rows)]
        changes = second["state"]["changes"]
        diff = changes[0].get("diff") or changes[0].get("input", "") if changes else ""
        new = make_row(
            first["state"]["task"], diff, False, 0, "none",
            source="cross-instance", validated="synthetic",
        )
        if new is not None:
            out.append(new)
    return out


def from_adversarial(limit: int) -> list[dict]:
    rows = []
    try:
        items = stream("mattymchen/codejudgebench-adversarial", "gemini_2.5_pro", config="codegen")
    except Exception as error:
        print(f"  adversarial unavailable: {error}", flush=True)
        return rows
    for item in items:
        task = f"{item.get('question_title', '')}\n\n{item.get('question_content', '')}"
        for key, req, bug in (("pos_response", True, 0), ("neg_response", False, 3)):
            row = make_row(
                task, as_diff(item.get(key) or ""), req, bug,
                "none" if req else "other",
                source="codejudge-adversarial", validated="dataset-execution",
            )
            if row is not None:
                rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def write(rows: list[dict], path: str) -> None:
    with open(path, "w") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    print(f"  {path}: {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="do not stream Hugging Face sources")
    parser.add_argument("--limit", type=int, default=0, help="cap each new source (0 = full)")
    parser.add_argument("--base-limit", type=int, default=0, help="cap each base m5f split (0 = full)")
    parser.add_argument("--repeat", type=int, default=2, help="repeat new train rows")
    parser.add_argument("--swe-extra", type=int, default=6000)
    parser.add_argument("--r2e", type=int, default=8000)
    parser.add_argument("--codejudge", type=int, default=6000)
    parser.add_argument("--perturb", type=int, default=2000)
    parser.add_argument("--cross", type=int, default=2000)
    parser.add_argument("--adversarial", type=int, default=1000)
    args = parser.parse_args()

    def cap(value: int) -> int:
        return min(value, args.limit) if args.limit else value

    print("base:")
    base = {split: load_base(split, args.base_limit) for split in ("train", "dev", "test")}
    for split, rows in base.items():
        print(f"  {split}_m5f: {len(rows)}")

    new: list[dict] = []
    if not args.offline:
        print("new sources:")
        swe = from_swe_bench_extra(cap(args.swe_extra))
        print(f"  swe-bench-extra: {len(swe)}")
        r2e = from_r2e(cap(args.r2e))
        print(f"  r2e-gym: {len(r2e)}")
        judge = from_codejudge(cap(args.codejudge))
        print(f"  codejudge: {len(judge)}")
        new.extend(swe)
        new.extend(r2e)
        new.extend(judge)
        new.extend(perturb_rows(swe + r2e, cap(args.perturb)))
        new.extend(cross_rows(swe + r2e, cap(args.cross)))
        ood = from_adversarial(cap(args.adversarial))
        print(f"  adversarial (OOD only): {len(ood)}")
        write(ood, os.path.join(BASE, "hard_ood_m6.jsonl"))

    train = list(base["train"])
    dev = list(base["dev"])
    test = list(base["test"])
    added = {"train": 0, "dev": 0, "test": 0}
    for row in new:
        split = row.get("split", "train")
        if split not in added:
            split = "train"
        target = train if split == "train" else dev if split == "dev" else test
        repeats = args.repeat if split == "train" else 1
        for _ in range(repeats):
            target.append(row)
        added[split] += repeats
    print(f"added: {added}")

    write(train, os.path.join(BASE, "train_m6.jsonl"))
    write(dev, os.path.join(BASE, "dev_m6.jsonl"))
    write(test, os.path.join(BASE, "test_m6.jsonl"))


if __name__ == "__main__":
    main()
