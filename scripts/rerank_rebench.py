"""N-best patch selection: does Laya pick the patch that actually passes?

This is the one shape where a bug detector pays off directly: instead of vetoing
one patch (which cannot fix anything), ask the model to choose among several
candidate patches for the same task. SWE-Rebench already ships exactly that:
3,855 tasks with >=4 candidate diffs and execution-verified labels.

    python scripts/rerank_rebench.py --limit 300 --k-max 12

Labels come from the dataset (`correct`), so no agent runs and no docker.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict

INSTRUCTIONS = "Several candidate patches fix the same issue. Which one is most likely correct?"
MAX_STATE = 900
POINTWISE = "Does this patch fully satisfy the issue description without breaking other behaviour?"

FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.M)
HUNK_RE = re.compile(r"^@@", re.M)


def added_lines(diff: str, limit: int = 2) -> list[str]:
    lines = [
        line[1:].strip() for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    ]
    return [line[:70] for line in lines[:limit]]


def synopsis(diff: str, rich: bool = False) -> tuple[str, dict]:
    files = FILE_RE.findall(diff)
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    hunks = len(HUNK_RE.findall(diff))
    short = ", ".join(files[:2])
    if len(files) > 2:
        short += f", +{len(files) - 2} more"
    text = f"+{added}/-{removed} in {len(files)}f ({hunks}h): {short}"
    if rich:
        examples = added_lines(diff)
        if examples:
            text += " | " + " ; ".join(examples)
    return text[:200], {"files": files, "added": added, "removed": removed, "hunks": hunks}


def ask(base_url: str, state: dict, criteria: dict, instructions: str = INSTRUCTIONS) -> dict[str, float]:
    body = {"state": state, "questions": {"pick": {
        "type": "choice", "instructions": instructions, "criteria": criteria}}}
    request = urllib.request.Request(
        f"{base_url}/v1/systemone", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())["answers"]["pick"].get("probabilities") or {}


def ask_noul(base_url: str, state: dict) -> float:
    body = {"state": state, "questions": {"req": {"type": "noul", "instructions": POINTWISE}}}
    request = urllib.request.Request(
        f"{base_url}/v1/systemone", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return float(json.loads(response.read())["answers"]["req"].get("noul") or 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/laya-ft/cache3/swe_rebench_traj.jsonl")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rich", action="store_true", help="add the first added lines to each option")
    parser.add_argument("--framing", default="correct", choices=("correct", "requirements"))
    parser.add_argument("--instructions", default="")
    parser.add_argument("--k-exact", type=int, default=0, help="only tasks with exactly this many candidates")
    parser.add_argument("--pointwise", action="store_true", help="score each candidate separately with a noul requirements_met question")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    by_task: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.data):
        row = json.loads(line)
        by_task[str(row["task"])[:MAX_STATE]].append(row)
    if args.k_exact:
        eligible = [(task, rows) for task, rows in by_task.items() if len(rows) == args.k_exact]
    else:
        eligible = [
            (task, rows) for task, rows in by_task.items()
            if args.k_min <= len(rows) <= args.k_max
        ]
    print(f"tasks total={len(by_task)} eligible={len(eligible)}", flush=True)
    if not eligible:
        return 1
    random.Random(args.seed).shuffle(eligible)
    sample = eligible[:args.limit]

    instructions = args.instructions or (
        "Several candidate patches address the same issue. Which patch most completely "
        "satisfies the issue description without breaking other behaviour?"
        if args.framing == "requirements" else INSTRUCTIONS
    )
    picks = 0
    hits = 0
    base = 0.0
    first_hits = 0
    smallest_hits = 0
    largest_hits = 0
    testfile_hits = 0
    margins = []
    started = time.time()
    details = []
    for index, (task, rows) in enumerate(sample, 1):
        candidates = [row["diff"] for row in rows]
        labeled = [bool(row["correct"]) for row in rows]
        metas = []
        criteria = {}
        for position, diff in enumerate(candidates):
            text, meta = synopsis(diff, args.rich)
            metas.append(meta)
            criteria[f"p{position + 1}"] = text
        try:
            if args.pointwise:
                scores = []
                for diff in candidates:
                    pw_state = {
                        "task": task[:350],
                        "changes": [{"tool": "read", "path": "patch.diff", "output": diff[:1200]}],
                    }
                    scores.append(ask_noul(args.base_url, pw_state))
                probabilities = {f"p{position + 1}": score for position, score in enumerate(scores)}
            else:
                state = {"task": task[:MAX_STATE], "changes": []}
                probabilities = ask(args.base_url, state, criteria, instructions)
        except Exception as error:  # noqa: BLE001 - keep the run alive
            print(f"  [{index}] {error}", flush=True)
            continue
        ordered = sorted(probabilities.items(), key=lambda row: -row[1])
        chosen = int(ordered[0][0][1:]) - 1
        picks += 1
        hits += int(labeled[chosen])
        base += sum(labeled) / len(labeled)
        first_hits += int(labeled[0])
        smallest_hits += int(labeled[min(range(len(metas)), key=lambda i: metas[i]["added"] + metas[i]["removed"])])
        largest_hits += int(labeled[max(range(len(metas)), key=lambda i: metas[i]["added"] + metas[i]["removed"])])
        testfile_hits += int(labeled[min(range(len(metas)), key=lambda i: any("test" in f for f in metas[i]["files"]))])
        correct_mass = sum(
            float(probability) for key, probability in probabilities.items()
            if labeled[int(key[1:]) - 1]
        )
        margins.append(correct_mass)
        details.append({
            "k": len(candidates), "chosen": chosen, "correct": labeled[chosen],
            "base_rate": round(sum(labeled) / len(labeled), 3),
            "correct_mass": round(correct_mass, 3),
            "pick": ordered[0][0], "pick_prob": round(ordered[0][1], 3),
        })
        if index % 25 == 0:
            print(f"  [{index}/{len(sample)}] pick@1={hits / picks:.3f} base={base / picks:.3f}", flush=True)

    n = max(1, picks)
    summary = {
        "tasks": picks,
        "pick_at_1": round(hits / n, 3),
        "random_baseline": round(base / n, 3),
        "delta": round(hits / n - base / n, 3),
        "first_candidate": round(first_hits / n, 3),
        "smallest_diff": round(smallest_hits / n, 3),
        "largest_diff": round(largest_hits / n, 3),
        "test_file_heuristic": round(testfile_hits / n, 3),
        "mean_correct_mass": round(sum(margins) / n, 3),
        "seconds": round(time.time() - started, 1),
        "k_mean": round(sum(row["k"] for row in details) / n, 1),
        "instructions": instructions,
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"summary": summary, "details": details}, handle, indent=1)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
