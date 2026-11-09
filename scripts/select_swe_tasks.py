"""Pick 20 diverse, hard SWE-smith instances (>=2 FAIL_TO_PASS, one per repo)."""
import json
import os

from datasets import load_dataset

OUT = os.path.join(os.environ.get("LAYACD_DATA", "/data/laya-ft"), "swe-smith", "tasks.json")
TARGET = 20
MIN_F2P = 2

ds = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)
seen = set()
tasks = []
for row in ds:
    repo = row["repo"]
    if repo in seen:
        continue
    if len(row["FAIL_TO_PASS"]) < MIN_F2P:
        continue
    if not row.get("image_name") or not row.get("problem_statement"):
        continue
    seen.add(repo)
    tasks.append(
        {
            "instance_id": row["instance_id"],
            "repo": repo,
            "image_name": row["image_name"],
            "problem_statement": row["problem_statement"],
            "FAIL_TO_PASS": list(row["FAIL_TO_PASS"]),
            "PASS_TO_PASS": list(row["PASS_TO_PASS"]),
        }
    )
    if len(tasks) >= TARGET:
        break

with open(OUT, "w") as f:
    json.dump(tasks, f, indent=2)
print(f"wrote {len(tasks)} tasks to {OUT}")
for t in tasks:
    print(f"  {t['instance_id']}  f2p={len(t['FAIL_TO_PASS'])} p2p={len(t['PASS_TO_PASS'])}  {t['image_name']}")
