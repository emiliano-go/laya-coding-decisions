"""Combined M2-M5 decision dataset v2.

Changes vs fetch_dec.py:
  - questions aligned with the runtime mutation set after retiring unsafe_patterns:
    requirements_met + bug_risk only;
  - patches stay in `changes[].input` (the field the model was trained on);
  - ordinal bug_risk gold (0 correct / 3 incorrect-or-perturbed), no synthetic level 2;
  - AXIOM-style perturbation hard negatives: correct patches with a rule-guided bug
    injected -> requirements_met true + bug_risk 3 (the buggy-guard class);
    cross-instance negatives (another task's correct patch) -> requirements_met false
    + bug_risk 0 (the unrelated-changes class);
  - NO synthesized execution evidence: the perfectly correlated evidence field
    taught the first retrain a shortcut instead of reading patches (probe: evidence
    fail -> bug 0.976, pass -> 0.33, absent -> constant). Execution failures are
    owned by the deterministic verify lane; the judge must read the diff;
  - v5 (SUFFIX m5d): task-derived memory context, label-independent. A
    requirement/constraint memory restates the task and a shared generic gotcha
    may be added; the choice is a function of the task hash only, so the same
    memory text appears across correct / perturbed / wrong-task rows and ~20% of
    rows carry an empty `memory: []`. Memory leads the state (after task) so the
    judge's token budget does not truncate it;
  - grouped instance split by task hash (train/dev/test) instead of random rows;
  - code-inspection rows from train_multi.jsonl are merged into train and a slice
    is held out for regression;
  - M3 harness labels are merged into dev.
Outputs: $LAYACD_DATA/{train,dev,test}_m5d.jsonl (default /data/laya-ft).
"""
import collections
import hashlib
import json
import os
import random

random.seed(23)
OUT = os.environ.get("LAYACD_DATA", "/data/laya-ft")
SUFFIX = "m5d"
CACHE = os.path.join(OUT, "cache3")
MAX_CHARS = 4000

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
}

# Task-independent memories. Selected deterministically from the task hash, so a
# given task gets the same memory text on every row it produces. This is the
# label-independence guard: memory content never depends on `correct`.
GENERIC_MEMORIES = [
    ("gotcha", "off-by-one history", "Prior fixes here regressed on boundary conditions; check loop bounds and inclusive ranges."),
    ("gotcha", "operator edits are high risk", "Comparisons and arithmetic here were changed before and silently broke behavior."),
    ("invariant", "public contract is stable", "Callers depend on the existing return shape and error behavior."),
    ("constraint", "do not widen scope", "Stay limited to the requested behavior; unrelated reformatting is a regression."),
]


def noul(correct):
    p = 0.95 if correct else 0.05
    return {"probabilities": {"false": round(1 - p, 4), "true": round(p, 4)}, "label": 1 if correct else 0}


def score(level):
    other = round(0.1 / 3, 4)
    return {"probabilities": {str(i): (0.9 if i == level else other) for i in range(4)}, "label": level}


def split_of(task):
    bucket = int(hashlib.sha1(task.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 100
    if bucket < 5:
        return "dev"
    if bucket < 10:
        return "test"
    return "train"


def derive_memories(task):
    """Task-derived memory context, independent of whether the patch is correct.

    The requirement memory restates the task; any second memory is drawn from a
    fixed task-independent pool by task hash. ~20% of tasks get an empty list so
    the judge never learns that memory presence alone predicts a label.
    """
    digest = hashlib.sha1(task.encode("utf-8", "ignore")).hexdigest()
    mid = digest[:8]
    bucket = int(digest, 16)
    if bucket % 5 == 0:
        return []
    memories = [{"id": mid, "type": "constraint", "title": "Task requirement", "statement": task[:200]}]
    if bucket % 3 != 0:
        kind, title, statement = GENERIC_MEMORIES[bucket % len(GENERIC_MEMORIES)]
        memories.append({"id": mid + "b", "type": kind, "title": title, "statement": statement})
    return memories


def make_row(task, diff, correct, extra_gold=None, answer="applied the change"):
    task = (task or "").strip()[:600]
    diff = (diff or "").strip()[:MAX_CHARS]
    if not task or not diff:
        return None
    gold = {"requirements_met": noul(correct), "bug_risk": score(0 if correct else 3)}
    if extra_gold:
        gold.update(extra_gold)
    state = {
        # memory leads so the 512-token head-slice at serve time keeps it.
        "task": task,
        "memory": derive_memories(task),
        "answer": answer,
        "changes": [{"tool": "apply_patch", "input": diff, "output": "applied"}],
        "toolCalls": [{"tool": "apply_patch", "title": "patch", "status": "completed"}],
    }
    return {"state": state, "questions": QUESTIONS, "gold": gold, "split": split_of(task)}


MUTATIONS = [
    ("==", "="),
    ("<=", "<"),
    (">=", ">"),
    ("!=", "=="),
    ("+ 1", "+ 2"),
    ("- 1", "- 2"),
    ("return True", "return False"),
    ("return False", "return True"),
    ("min(", "max("),
    ("max(", "min("),
    ("sorted(", "list("),
    (" and ", " or "),
    (" + ", " - "),
]


def perturb(diff):
    lines = diff.split("\n")
    added = [i for i, line in enumerate(lines) if line.startswith("+") and not line.startswith("+++")]
    random.shuffle(added)
    for index in added:
        for old, new in MUTATIONS:
            if old in lines[index]:
                lines[index] = lines[index].replace(old, new, 1)
                return "\n".join(lines)
    return None


def load_cache(name):
    path = os.path.join(CACHE, name + ".jsonl")
    if not os.path.exists(path):
        print(f"  missing cache {path}", flush=True)
        return []
    rows = [json.loads(line) for line in open(path)]
    print(f"  {name}: {len(rows)}", flush=True)
    return rows


def main():
    sources = {
        "swe_smith_traj": load_cache("swe_smith_traj"),
        "swe_rebench_traj": load_cache("swe_rebench_traj"),
        "gold": load_cache("gold"),
        "bigcodebench": load_cache("bigcodebench"),
    }
    rows = []
    for name, source in sources.items():
        for r in source:
            row = make_row(r["task"], r["diff"], bool(r["correct"]))
            if row:
                rows.append(row)

    positives = [r for source in sources.values() for r in source if r["correct"]]
    print(f"  positives: {len(positives)}")

    # Cross-instance negatives: a task paired with another task's patch.
    # The patch itself is known-correct code, so bug_risk 0; only requirements fail.
    for i, r in enumerate(positives[:12000]):
        other = positives[(i + 1 + random.randrange(len(positives) - 1)) % len(positives)]
        row = make_row(r["task"], other["diff"], False, extra_gold={"bug_risk": score(0)})
        if row:
            # The patch is known-correct code; only the task requirement memory
            # reveals it answers a different task. Mark these for ablation eval.
            row["slice"] = "memory-ablation"
            rows.append(row)

    # AXIOM-style hard negatives: correct patch + rule-guided bug injection.
    perturbed = 0
    for r in positives:
        if perturbed >= 40000:
            break
        if random.random() > 0.55:
            continue
        mutated = perturb(r["diff"])
        if mutated is None:
            continue
        row = make_row(r["task"], mutated, True, extra_gold={"bug_risk": score(3)})
        if row:
            rows.append(row)
            perturbed += 1
    print(f"  perturbations: {perturbed}")

    # M3 harness labels -> dev.
    harness = []
    labels_path = os.path.join(OUT, "funnel-bench", "labels-m3.jsonl")
    if os.path.exists(labels_path):
        for line in open(labels_path):
            row = json.loads(line)
            state = row["state"] if isinstance(row["state"], dict) else json.loads(row["state"])
            questions = row["questions"] if isinstance(row["questions"], dict) else json.loads(row["questions"])
            gold = row["gold"] if isinstance(row["gold"], dict) else json.loads(row["gold"])
            gold = {k: v for k, v in gold.items() if k in QUESTIONS}
            if not gold:
                continue
            # Strip any captured execution evidence: the judge must not learn it.
            harness_state = {k: v for k, v in state.items() if k != "evidence"}
            harness.append({"state": harness_state, "questions": QUESTIONS, "gold": gold, "split": "dev"})
        rows.extend(harness)
    print(f"  harness labels: {len(harness)}")

    # Code-inspection rows from the multi-task mix (keep both abilities).
    inspection = []
    multi = os.path.join(OUT, "train_multi.jsonl")
    if os.path.exists(multi):
        for line in open(multi):
            row = json.loads(line)
            if "workflow" not in row:
                continue
            inspection.append(
                {
                    "state": row["state"],
                    "questions": row["questions"],
                    "gold": row["gold"],
                    "split": "dev" if random.random() < 0.02 else "train",
                }
            )
    print(f"  inspection rows: {len(inspection)}")

    for i, row in enumerate(rows):
        row["id"] = f"m5-{i:07d}"
    for i, row in enumerate(inspection):
        row["id"] = f"in-{i:07d}"

    def dump(split, path):
        with open(path, "w") as f:
            for row in rows + inspection:
                if row["split"] != split:
                    continue
                f.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "state": row["state"],
                            "questions": row["questions"] if isinstance(row["questions"], str) else json.dumps(row["questions"]),
                            "gold": row["gold"] if isinstance(row["gold"], str) else json.dumps(row["gold"]),
                            **({"workflow": "inspection"} if row["id"].startswith("in-") else {}),
                            **({"slice": row["slice"]} if row.get("slice") else {}),
                        }
                    )
                    + "\n"
                )

    dump("train", os.path.join(OUT, f"train_{SUFFIX}.jsonl"))
    dump("dev", os.path.join(OUT, f"dev_{SUFFIX}.jsonl"))
    dump("test", os.path.join(OUT, f"test_{SUFFIX}.jsonl"))

    stats = collections.Counter()
    labels = collections.defaultdict(collections.Counter)
    for row in rows + inspection:
        stats[row["split"]] += 1
        gold = row["gold"]
        if isinstance(gold, str):
            gold = json.loads(gold)
        for qid, entry in gold.items():
            labels[qid][entry["label"]] += 1
    print("\nsplits:", dict(stats))
    for qid, counter in labels.items():
        print(f"  {qid}: {dict(counter)}")


if __name__ == "__main__":
    main()
