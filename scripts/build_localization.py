"""Build hunk-localization choice rows from deterministic mutation rows.

Each row asks: which diff hunk contains the injected bug? The gold hunk is
found by comparing the mutated patch with the original patch for the same task.
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize_state import normalize_state, task_split  # noqa: E402

BASE = os.environ.get("LAYACD_DATA", "/data/laya-ft")
QUESTIONS = {
    "loc": {
        "type": "choice",
        "instructions": "Which diff hunk most likely contains the bug described by the task?",
    },
}


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


def gold_hunk(mutated: str, original: str) -> int | None:
    base = {line for line in original.splitlines() if line.startswith("+") and not line.startswith("+++")}
    for index, hunk in enumerate(split_hunks(mutated)):
        for line in added(hunk):
            if "+" + line not in base:
                return index
    return None


def choice_gold(label: int, count: int) -> dict:
    other = round(0.1 / max(1, count - 1), 4)
    return {
        "probabilities": {f"hunk-{i}": (0.9 if i == label else other) for i in range(count)},
        "label": f"hunk-{label}",
    }


def main() -> None:
    base: dict[str, str] = {}
    mutated: list[tuple[str, str]] = []
    for line in open(os.path.join(BASE, "train_v7.jsonl")):
        row = json.loads(line)
        state = row["state"]
        changes = state.get("changes") or []
        diff = changes[0].get("input", "") if changes else ""
        if not diff:
            continue
        if str(row.get("source", "")).startswith("mutation-"):
            mutated.append((state["task"], diff))
        else:
            base.setdefault(state["task"], diff)

    random.seed(23)
    rows = []
    for task, diff in mutated:
        original = base.get(task)
        if original is None:
            continue
        hunks = split_hunks(diff)
        if len(hunks) < 2:
            continue
        label = gold_hunk(diff, original)
        if label is None:
            continue
        criteria = {}
        for index, hunk in enumerate(hunks):
            text = " / ".join(line.strip() for line in added(hunk)[:4])[:160] or hunk[0]
            criteria[f"hunk-{index}"] = text
        state = normalize_state(
            task=task,
            answer="applied the change",
            changes=[{"tool": "apply_patch", "input": diff, "output": "applied"}],
            tool_calls=[{"tool": "apply_patch", "title": "patch", "status": "completed"}],
            grounding=[],
            memory=[],
        )
        rows.append({
            "state": state,
            "questions": json.dumps({"loc": {**QUESTIONS["loc"], "criteria": criteria}}),
            "gold": json.dumps({"loc": choice_gold(label, len(hunks))}),
            "split": "dev" if random.random() < 0.1 else "test" if random.random() < 0.05 else "train",
            "source": "loc-mutation",
        })

    for split in ("train", "dev", "test"):
        subset = [row for row in rows if row["split"] == split]
        path = os.path.join(BASE, f"loc_{split}.jsonl")
        with open(path, "w") as sink:
            for row in subset:
                sink.write(json.dumps(row) + "\n")
        print(f"{path}: {len(subset)}")
    print(f"total {len(rows)}")


if __name__ == "__main__":
    main()
