"""Summarize a run_swe_smith results.jsonl by condition."""
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines() if line.strip()]
by = defaultdict(list)
for row in rows:
    by[row["condition"]].append(row)

print(f"{'condition':12s} {'n':>3s} {'pass':>4s} {'fail':>4s} {'invalid':>7s} {'steps':>6s} {'secs':>7s} {'review':>6s} {'flips':>5s}")
off = {r["instance_id"]: r["passed"] for r in by.get("off", []) if r.get("passed") is not None}
for cond, items in by.items():
    valid = [r for r in items if r.get("invalid") is None and r.get("passed") is not None]
    passed = sum(1 for r in valid if r["passed"])
    steps = sum((r.get("coverage") or {}).get("step", 0) for r in valid) / max(1, len(valid))
    secs = sum(r.get("seconds") or 0 for r in valid) / max(1, len(valid))
    reviews = 0
    for r in valid:
        log = root / r["instance_id"] / f"decision-{cond}" / "decision-log.jsonl"
        if log.exists():
            last = json.loads(log.read_text().splitlines()[-1])
            reviews += 1 if last.get("reviewOnly") else 0
    flips = sum(1 for r in valid if r["instance_id"] in off and off[r["instance_id"]] != r["passed"])
    print(f"{cond:12s} {len(valid):3d} {passed:4d} {len(valid)-passed:4d} {len(items)-len(valid):7d} {steps:6.1f} {secs:7.1f} {reviews:6d} {flips:5d}")
