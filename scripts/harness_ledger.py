"""Import harness results into a durable Totem ledger.

Each valid (instance, condition) row becomes one observation memory in a
dedicated project, tagged so it can be recalled with ordinary memory search.
Rows already imported are skipped using a small sidecar state file.

  python scripts/harness_ledger.py --ledger /data/laya-ft/harness-ledger \
    --root /data/laya-ft/swe-smith-memory --root /data/laya-ft/swe-smith-ab-valid
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

TOTEM = os.environ.get("TOTEM_BIN") or shutil.which("totem") or os.path.expanduser("~/.local/bin/totem")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def rows_in(root):
    path = os.path.join(root, "results.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="/data/laya-ft/harness-ledger")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    roots = args.root or ["/data/laya-ft/swe-smith-memory", "/data/laya-ft/swe-smith-ab-valid", "/data/laya-ft/swe-smith-confirm"]
    state_path = os.path.join(args.ledger, "imported.json")
    imported = set()
    if os.path.exists(state_path):
        imported = set(json.load(open(state_path)))
    if not args.dry_run:
        os.makedirs(args.ledger, exist_ok=True)
        run([TOTEM, "--project", args.ledger, "init"])

    created = 0
    skipped = 0
    for root in roots:
        for row in rows_in(root):
            if row.get("invalid") is not None or row.get("passed") is None:
                skipped += 1
                continue
            key = f"{row['instance_id']}|{row['condition']}"
            if key in imported:
                skipped += 1
                continue
            repo = str(row.get("repo") or row["instance_id"].split(".")[0]).split("/")[-1]
            verdict = "pass" if row["passed"] else "fail"
            cov = row.get("coverage") or {}
            statement = (
                f"SWE-smith {row['instance_id']} [{row['condition']}] {verdict} in "
                f"{row.get('seconds')}s ({cov.get('step', 0)} decision steps, "
                f"memory calls {row.get('memory_calls', 0)}). Log: {row.get('agent_log')}"
            )
            title = f"SWE-smith {repo} [{row['condition']}] {verdict}"
            tags = f"harness:swe,condition:{row['condition']},repo:{repo},verdict:{verdict}"
            print(f"{'[dry] ' if args.dry_run else ''}{title}")
            if args.dry_run:
                imported.add(key)
                created += 1
                continue
            proc = run([
                TOTEM, "--project", args.ledger, "create", "--type", "observation",
                "--title", title, "--statement", statement, "--tags", tags,
                "--asserted-by", "test", "--metadata", '{"observation": "harness run result"}',
            ])
            if proc.returncode != 0:
                print(f"  failed: {proc.stderr.strip()[:200]}")
                continue
            imported.add(key)
            created += 1
    if not args.dry_run:
        os.makedirs(args.ledger, exist_ok=True)
        json.dump(sorted(imported), open(state_path, "w"), indent=2)
    print(f"created={created} skipped={skipped} ledger={args.ledger}")


if __name__ == "__main__":
    main()
