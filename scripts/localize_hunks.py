"""Zero-shot hunk localization probe.

Takes deterministic mutation rows (correct patch + one injected mutation), finds
the mutated hunk by comparing with the original patch, and asks Laya to rank the
hunks. This is the "pin down the bad hunk" step; no training.

  python scripts/localize_hunks.py --limit 100
"""
from __future__ import annotations

import argparse
import json
import urllib.request


def split_hunks(diff: str) -> list[list[str]]:
    hunks: list[list[str]] = []
    current: list[str] = []
    for line in diff.splitlines():
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        hunks.append(current)
    return hunks


def added(lines: list[str]) -> list[str]:
    return [line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")]


def mutated_hunk(mutated: str, base: str) -> int | None:
    base_added = {line for line in base.splitlines() if line.startswith("+") and not line.startswith("+++")}
    hunks = split_hunks(mutated)
    for index, hunk in enumerate(hunks):
        for line in added(hunk):
            if "+" + line not in base_added:
                return index
    return None


def ask(base_url: str, task: str, diff: str, hunks: list[list[str]]) -> dict[str, float]:
    criteria = {}
    for index, hunk in enumerate(hunks):
        text = " / ".join(line.strip() for line in added(hunk)[:3])[:120] or hunk[0][:120]
        criteria[f"hunk-{index}"] = text
    body = json.dumps({
        "state": {
            "task": task,
            "changes": [{"tool": "apply_patch", "input": diff, "output": "applied"}],
        },
        "questions": {
            "loc": {
                "type": "choice",
                "instructions": "Which diff hunk most likely contains the bug or incorrect edge case?",
                "criteria": criteria,
            },
        },
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/systemone", data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = json.loads(response.read())
    return payload["answers"]["loc"]["probabilities"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/laya-ft/train_v7.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    base: dict[str, str] = {}
    mutated: list[dict] = []
    for line in open(args.data):
        row = json.loads(line)
        source = str(row.get("source", ""))
        state = row["state"]
        changes = state.get("changes") or []
        diff = changes[0].get("input", "") if changes else ""
        if not diff:
            continue
        if source.startswith("mutation-"):
            mutated.append({"task": state["task"], "diff": diff})
        else:
            base.setdefault(state["task"], diff)

    ranks = []
    rows = []
    for row in mutated:
        original = base.get(row["task"])
        if original is None:
            continue
        hunks = split_hunks(row["diff"])
        if len(hunks) < 2:
            continue
        gold = mutated_hunk(row["diff"], original)
        if gold is None:
            continue
        try:
            probabilities = ask(args.base_url, row["task"], row["diff"], hunks)
        except Exception as error:
            print("ask failed", error)
            continue
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        rank = next((i for i, (key, _) in enumerate(ordered, 1) if key == f"hunk-{gold}"), None)
        if rank is None:
            continue
        ranks.append(rank)
        rows.append({"task": row["task"][:80], "gold": gold, "rank": rank, "hunks": len(hunks), "top": ordered[0][0], "top_prob": ordered[0][1]})
        print(f"[{len(ranks)}/{args.limit}] rank {rank}/{len(hunks)}", flush=True)
        if len(ranks) >= args.limit:
            break

    n = len(ranks)
    print(json.dumps({
        "evaluated": n,
        "top1": round(sum(1 for r in ranks if r == 1) / max(1, n), 3),
        "top2": round(sum(1 for r in ranks if r <= 2) / max(1, n), 3),
        "mrr": round(sum(1 / r for r in ranks) / max(1, n), 3),
        "rows": rows[:50],
    }, indent=2))


if __name__ == "__main__":
    main()
