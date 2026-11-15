"""Decision dataset v7: better labels for bug_class, scope, and graded risk.

Changes vs v6:
  - one four-question set: requirements_met, no_unrelated_changes, bug_risk,
    bug_class (so scope has its own head);
  - teacher rows keep their graded bug_risk (1/2/3) and add a
    `no_unrelated_changes` gold (already present) plus `scope-unrelated` on
    scope rows;
  - teacher preference pairs supply balanced `bug_class` labels from the
    `dimensions.defect` field (wrong-comparison, off-by-one, null-empty,
    type-coercion, resource-leak, swallowed-exception, other);
  - deterministic class mutations fill underrepresented classes;
  - deterministic scope negatives add an unrelated file hunk to a correct
    patch (requirements_met=1, bug_risk=0, no_unrelated_changes=0);
  - split remains grouped by task hash.

Usage:
  python scripts/gen_dataset_v7.py --base /data/laya-ft/train_m6s.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

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
    "no_unrelated_changes": {
        "type": "noul",
        "instructions": "The code changes contain no modifications unrelated to the user's request.",
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

# Teacher defect strings -> runtime bug_class keys.
DEFECT_MAP = {
    "": "none",
    "none": "none",
    "off-by-one": "off-by-one",
    "wrong-operator": "wrong-comparison",
    "wrong-comparison": "wrong-comparison",
    "null/empty-handling": "null-empty",
    "null-empty": "null-empty",
    "type-coercion": "type-coercion",
    "resource-leak": "resource-leak",
    "wrong-exception-scope": "swallowed-exception",
    "swallowed exception": "swallowed-exception",
    "swallowed-exception": "swallowed-exception",
    "early-return": "other",
    "broken-invariant": "other",
    "wrong-argument": "other",
    "wrong-return": "other",
}

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
    ("Number(", "String(", "type-coercion"),
    ("parseInt(", "parseFloat(", "type-coercion"),
    ("=== ", "== ", "type-coercion"),
]


def choice_gold(label: str) -> dict:
    keys = list(BUG_CLASS)
    other = round(0.1 / (len(keys) - 1), 4)
    return {
        "probabilities": {key: (0.9 if key == label else other) for key in keys},
        "label": keys.index(label),
    }


def gold(requirements: bool, unrelated_ok: bool, risk: int, bug_class: str | None) -> dict:
    out = {
        "requirements_met": noul(requirements),
        "no_unrelated_changes": noul(unrelated_ok),
        "bug_risk": score(max(0, min(3, risk))),
    }
    if bug_class is not None:
        out["bug_class"] = choice_gold(bug_class)
    return out


def make_row(
    task: str,
    diff: str,
    gold_value: dict,
    *,
    source: str,
    split: str | None = None,
) -> dict | None:
    diff = (diff or "").strip()[:MAX_CHARS]
    if not task.strip() or not diff:
        return None
    state = normalize_state(
        task=task,
        answer="applied the change",
        changes=[{"tool": "apply_patch", "input": diff, "output": "applied"}],
        tool_calls=[{"tool": "apply_patch", "title": "patch", "status": "completed"}],
        grounding=[],
        memory=derive_memories(task),
    )
    return {
        "state": state,
        "questions": json.dumps(QUESTIONS),
        "gold": json.dumps(gold_value),
        "split": split or task_split(task),
        "source": source,
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
    task = state.get("task", "")
    changes = state.get("changes") if isinstance(state.get("changes"), list) else []
    row["state"] = normalize_state(
        task=task,
        answer=state.get("answer", ""),
        changes=changes,
        tool_calls=state.get("toolCalls") if isinstance(state.get("toolCalls"), list) else [],
        grounding=state.get("grounding") if isinstance(state.get("grounding"), list) else [],
        memory=state.get("memory") if isinstance(state.get("memory"), list) else [],
    )
    row["questions"] = json.dumps(QUESTIONS)
    if isinstance(row.get("gold"), dict):
        row["gold"] = json.dumps(row["gold"])
    return row


def load_base(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        row = normalize_row(json.loads(line))
        if row is not None:
            rows.append(row)
    return rows


def load_teacher(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        row = normalize_row(json.loads(line))
        if row is None:
            continue
        g = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        # Scope rows get an explicit class; positives get none. Bug rows keep
        # their graded risk but the exact class is unknown.
        if row.get("slice") == "teacher-scope":
            g["bug_class"] = choice_gold("scope-unrelated")
        elif row.get("slice") == "teacher-positive":
            g["bug_class"] = choice_gold("none")
        row["gold"] = json.dumps(g)
        row["source"] = "teacher"
        rows.append(row)
    return rows


def load_pairs(path: str, cap_per_class: int = 900) -> list[dict]:
    rows = []
    counts: Counter[str] = Counter()
    for line in open(path):
        pair = json.loads(line)
        task = pair.get("task", "")
        chosen = pair.get("chosen", "")
        rejected = pair.get("rejected", "")
        defect = (pair.get("dimensions") or {}).get("defect", "")
        cls = DEFECT_MAP.get(defect, "other")
        if not task or not chosen or not rejected:
            continue
        if counts[cls] >= cap_per_class:
            continue
        split = task_split(task)
        row = make_row(task, chosen, gold(True, True, 0, "none"), source="teacher-pair", split=split)
        if row is not None:
            rows.append(row)
        # An empty/none defect means the pair differed on approach/minimality,
        # not on correctness; the rejected patch is not a bug.
        if cls != "none":
            row = make_row(task, rejected, gold(True, True, 3, cls), source="teacher-pair", split=split)
            if row is not None:
                rows.append(row)
        counts[cls] += 1
    print("  pairs by class", dict(counts))
    return rows


def mutate(diff: str, kind: str) -> str | None:
    lines = diff.split("\n")
    added = [index for index, line in enumerate(lines) if line.startswith("+") and not line.startswith("+++")]
    random.shuffle(added)
    for index in added:
        for old, new, label in MUTATIONS:
            if label != kind:
                continue
            if old in lines[index]:
                lines[index] = lines[index].replace(old, new, 1)
                return "\n".join(lines)
    return None


def class_rows(base_rows: list[dict], per_class: int) -> list[dict]:
    out = []
    counts: Counter[str] = Counter()
    for row in base_rows:
        if counts["off-by-one"] >= per_class and counts["wrong-comparison"] >= per_class and counts["type-coercion"] >= per_class:
            break
        state = row["state"]
        changes = state.get("changes") or []
        diff = changes[0].get("input", "") if changes else ""
        if not diff:
            continue
        for kind in ("off-by-one", "wrong-comparison", "type-coercion"):
            if counts[kind] >= per_class:
                continue
            mutated = mutate(diff, kind)
            if mutated is None:
                continue
            new = make_row(state["task"], mutated, gold(True, True, 3, kind), source=f"mutation-{kind}", split=row.get("split"))
            if new is not None:
                out.append(new)
                counts[kind] += 1
    print("  mutation classes", dict(counts))
    return out


def scope_rows(base_rows: list[dict], limit: int) -> list[dict]:
    out = []
    for row in base_rows[:limit]:
        state = row["state"]
        changes = state.get("changes") or []
        diff = changes[0].get("input", "") if changes else ""
        if not diff:
            continue
        unrelated = diff + "\n--- a/README.md\n+++ b/README.md\n@@ -0,0 +1,2 @@\n+Unrelated documentation update.\n+Another unrelated line.\n"
        new = make_row(state["task"], unrelated, gold(True, False, 0, "scope-unrelated"), source="scope-deterministic", split=row.get("split"))
        if new is not None:
            out.append(new)
    print("  deterministic scope", len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.join(BASE, "train_m6s.jsonl"))
    parser.add_argument("--out-prefix", default="v7")
    parser.add_argument("--class-cap", type=int, default=900)
    parser.add_argument("--mutations", type=int, default=4000)
    parser.add_argument("--scopes", type=int, default=3000)
    args = parser.parse_args()

    print("base", args.base)
    train = load_base(args.base)
    dev = load_base(os.path.join(BASE, "dev_m6.jsonl"))
    test = load_base(os.path.join(BASE, "test_m6.jsonl"))
    print(f"  base train/dev/test {len(train)}/{len(dev)}/{len(test)}")

    teacher = load_teacher(os.path.join(BASE, "gen_deepseek", "teacher_all.jsonl"))
    pairs = load_pairs(os.path.join(BASE, "gen_deepseek", "pairs_teacher.jsonl"), args.class_cap)
    mutations = class_rows(train, args.mutations // 3)
    scopes = scope_rows(train, args.scopes)
    extra = teacher + pairs + mutations + scopes
    print(f"  teacher {len(teacher)} pairs {len(pairs)} mutations {len(mutations)} scopes {len(scopes)}")

    added = {"train": 0, "dev": 0, "test": 0}
    for row in extra:
        split = row.get("split", "train")
        target = train if split == "train" else dev if split == "dev" else test
        target.append(row)
        added[split] += 1
    print(f"  added {added}")

    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        path = os.path.join(BASE, f"{name}_{args.out_prefix}.jsonl")
        with open(path, "w") as sink:
            for row in rows:
                sink.write(json.dumps(row) + "\n")
        print(f"  {path}: {len(rows)}")
        classes = Counter()
        for row in rows:
            g = json.loads(row["gold"])
            if "bug_class" in g:
                classes[g["bug_class"]["label"]] += 1
        print("    bug_class", dict(classes))


if __name__ == "__main__":
    main()
