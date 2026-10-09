"""Build an encoder-style decision dataset with test-verified labels.

Sources:
  A. resolved-labeled agent trajectories (SWE-smith, SWE-rebench openhands)
  B. gold-patch instances (SWE-Gym, SWE-smith)
  C. function-level canonical vs deterministic mutant (BigCodeBench)
  D. security (BigVul) for unsafe_patterns
  plus synthesized no_unrelated_changes / unsafe_patterns / cross-instance rows.

Questions mirror encoder's DEFAULT_QUESTIONS exactly. Output:
/data/laya-ft/train_dec.jsonl + dev_dec.jsonl.
"""
import hashlib
import json
import os
import random

random.seed(17)
OUT = os.environ.get("LAYACD_DATA", "/data/laya-ft")
CACHE = os.path.join(OUT, "cache3")
MAX_CHARS = 1500
TARGET = 320_000

QUESTIONS = {
    "requirements_met": {
        "type": "noul",
        "instructions": "The code changes fully satisfy the user's request described in the task.",
    },
    "no_unrelated_changes": {
        "type": "noul",
        "instructions": "The code changes contain no modifications unrelated to the user's request.",
    },
    "unsafe_patterns": {
        "type": "noul",
        "instructions": "The code changes introduce hardcoded secrets or credentials, SQL or shell injection sinks, or unsafe deserialization.",
    },
    "bug_risk": {
        "type": "score",
        "instructions": "Rate the risk that the code changes introduce bugs.",
        "criteria": [
            "No risk: changes are trivially correct",
            "Low risk: minor logic changes, easily verified",
            "Medium risk: non-trivial logic with plausible edge-case bugs",
            "High risk: likely broken or clearly incorrect logic",
        ],
    },
}

ROWS = []
BY_KEY = {}


def noul(correct):
    p = 0.95 if correct else 0.05
    return {"probabilities": {"false": round(1 - p, 4), "true": round(p, 4)}, "label": 1 if correct else 0}


def score(level, n=4):
    level = max(0, min(level, n - 1))
    other = round(0.1 / max(1, n - 1), 4)
    return {"probabilities": {str(i): (0.9 if i == level else other) for i in range(n)}, "label": level}


def emit(task, diff, labels, answer="applied the change"):
    """Create or merge a row for (task, diff) with the given question labels."""
    task = (task or "").strip()
    diff = (diff or "").strip()
    if not task or not diff or not labels:
        return
    task = task[:400]
    diff = diff[: MAX_CHARS - len(task)]
    key = hashlib.sha1((task + diff).encode("utf-8", "ignore")).hexdigest()
    row = BY_KEY.get(key)
    if row is None:
        row = {
            "state": {
                "task": task,
                "answer": answer,
                "changes": [{"tool": "apply_patch", "input": diff, "output": "applied"}],
                "toolCalls": [{"tool": "apply_patch", "title": "patch", "status": "completed"}],
            },
            "questions": {},
            "gold": {},
        }
        BY_KEY[key] = row
        ROWS.append(row)
    for qid, label in labels.items():
        row["questions"][qid] = QUESTIONS[qid]
        row["gold"][qid] = label


def balanced_source(name, produce, cap):
    path = os.path.join(CACHE, name + ".jsonl")
    if os.path.exists(path):
        rows = [json.loads(line) for line in open(path)]
        print(f"  {name}: cached {len(rows)}", flush=True)
        return rows
    seen = set()
    rows = []
    counts = {True: 0, False: 0}
    for task, diff, correct, extra in produce():
        if counts[True] >= cap // 2 and counts[False] >= cap // 2:
            break
        if counts[correct] >= cap // 2:
            continue
        key = hashlib.sha1((task + diff).encode("utf-8", "ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"task": task, "diff": diff, "correct": correct, "extra": extra})
        counts[correct] += 1
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {name}: {len(rows)} ({counts[True]} true / {counts[False]} false)", flush=True)
    return rows


def first_user_text(messages):
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return messages[:MAX_CHARS]
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")[:MAX_CHARS]
    return ""


def src_swe_smith_traj():
    from datasets import load_dataset

    def produce():
        for row in load_dataset("SWE-bench/SWE-smith-trajectories", split="tool", streaming=True):
            task = first_user_text(row.get("messages"))
            patch = row.get("patch") or ""
            if task and patch:
                yield task, patch, bool(row.get("resolved")), "traj"
    return balanced_source("swe_smith_traj", produce, 60000)


def src_swe_rebench_traj():
    from datasets import load_dataset

    def produce():
        for row in load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train", streaming=True):
            task = first_user_text(row.get("trajectory"))
            patch = row.get("model_patch") or ""
            if task and patch:
                yield task, patch, bool(row.get("resolved")), "traj"
    return balanced_source("swe_rebench_traj", produce, 80000)


def src_gold_instances():
    from datasets import load_dataset

    def produce():
        for ds_name, split in (("SWE-Gym/SWE-Gym", "train"), ("SWE-bench/SWE-smith", "train")):
            try:
                ds = load_dataset(ds_name, split=split, streaming=True)
            except Exception as error:
                print(f"    {ds_name} unavailable: {error}")
                continue
            n = 0
            for row in ds:
                task = row.get("problem_statement") or ""
                patch = row.get("patch") or ""
                if task and patch:
                    yield task, patch, True, "gold"
                n += 1
                if n >= 30000:
                    break
    return balanced_source("gold", produce, 30000)


def src_bigcodebench():
    from datasets import load_dataset

    def mutants(code):
        out = []
        for old, new in (("+ 1", "+ 2"), ("<=", "<"), ("==", "!="), ("min(", "max("), ("sorted(", "reversed(")):
            if old in code:
                out.append(code.replace(old, new, 1))
        return out[:2]

    def produce():
        ds = load_dataset("bigcode/bigcodebench", "default", split="v0.1.4", streaming=True)
        n = 0
        for row in ds:
            task = (row.get("complete_prompt") or "")[:MAX_CHARS]
            canonical = row.get("canonical_solution") or ""
            if not task or not canonical:
                continue
            yield task, canonical, True, "canonical"
            for m in mutants(canonical):
                yield task, m, False, "mutant"
            n += 1
            if n >= 800:
                break
    return balanced_source("bigcodebench", produce, 6000)


def cached_rows(source):
    for base in (CACHE, os.path.join(OUT, "cache2")):
        path = os.path.join(base, source + ".jsonl")
        if os.path.exists(path):
            rows = [tuple(json.loads(line)) for line in open(path)]
            print(f"  {source}: cached {len(rows)}", flush=True)
            return rows
    return []


def main():
    traj = src_swe_smith_traj() + src_swe_rebench_traj()
    gold = src_gold_instances()
    function = src_bigcodebench()
    positives = [r for r in traj + gold + function if r["correct"]]
    print(f"  positives: {len(positives)}", flush=True)

    # requirements_met + bug_risk (the core judge objective)
    for r in traj + gold + function:
        level = random.choice([0, 0, 0, 1]) if r["correct"] else (2 if r["extra"] == "mutant" else 3)
        emit(r["task"], r["diff"], {"requirements_met": noul(r["correct"]), "bug_risk": score(level)})

    # cross-instance negatives: task paired with a different task's patch
    for i, r in enumerate(positives[:12000]):
        other = positives[(i + 1 + random.randrange(len(positives) - 1)) % len(positives)]
        emit(r["task"], other["diff"], {"requirements_met": noul(False), "bug_risk": score(3)})

    # no_unrelated_changes: balanced positives and unrelated-edit negatives
    unrelated = "\ndiff --git a/UNRELATED.md b/UNRELATED.md\n--- a/UNRELATED.md\n+++ b/UNRELATED.md\n@@ -1 +1 @@\n-old docs\n+new docs\n"
    sample = positives[:12000]
    for r in sample:
        emit(r["task"], r["diff"], {"no_unrelated_changes": noul(True)})
        emit(r["task"], r["diff"] + unrelated, {"no_unrelated_changes": noul(False)})

    # unsafe_patterns: safe positives and secret-injected positives, plus BigVul
    secret = '\n+API_KEY = "sk-live-9f8e7d6c5b4a3210"\n'
    for r in positives[:8000]:
        emit(r["task"], r["diff"], {"unsafe_patterns": noul(False)})
        emit(r["task"], r["diff"] + secret, {"unsafe_patterns": noul(True)})
    for state, statement, correct in cached_rows("vul")[:30000]:
        emit(statement[:MAX_CHARS], state, {"unsafe_patterns": noul(bool(correct))}, answer=statement[:200])

    # trim dedicated majority rows for the secondary questions to balance them,
    # keeping multi-label rows intact
    for qid in ("no_unrelated_changes", "unsafe_patterns"):
        by_label = {}
        for row in ROWS:
            g = row["gold"].get(qid)
            if g is not None:
                by_label.setdefault(g["label"], []).append(row)
        if len(by_label) < 2:
            continue
        n = min(len(v) for v in by_label.values())
        drop = set()
        for rows_ in by_label.values():
            if len(rows_) <= n:
                continue
            only = [r for r in rows_ if len(r["gold"]) == 1]
            multi = [r for r in rows_ if len(r["gold"]) > 1]
            keep = multi + only[: max(0, n - len(multi))]
            keep_ids = {id(r) for r in keep}
            for r in rows_:
                if id(r) not in keep_ids:
                    drop.add(id(r))
        ROWS[:] = [r for r in ROWS if id(r) not in drop]

    random.shuffle(ROWS)
    final = ROWS[:TARGET]
    for i, row in enumerate(final):
        row["id"] = f"dec-{i:07d}"
    dev = final[:2000]
    train = final[2000:]

    def dump(rows, path):
        with open(path, "w") as f:
            for row in rows:
                f.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "state": row["state"],
                            "questions": json.dumps(row["questions"]),
                            "gold": json.dumps(row["gold"]),
                        }
                    )
                    + "\n"
                )

    dump(train, os.path.join(OUT, "train_dec.jsonl"))
    dump(dev, os.path.join(OUT, "dev_dec.jsonl"))

    import collections

    counts = collections.Counter()
    labels = collections.defaultdict(collections.Counter)
    for row in final:
        for qid, g in row["gold"].items():
            counts[qid] += 1
            labels[qid][g["label"]] += 1
    print(f"\nwrote {len(train)} train / {len(dev)} dev")
    print("questions:", dict(counts))
    for qid, c in labels.items():
        print(f"  {qid}: {dict(c)}")


if __name__ == "__main__":
    main()
