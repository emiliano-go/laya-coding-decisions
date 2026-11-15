"""Canonical judge state, mirrored from dsh-totem `src/laya.ts`.

Training, evaluation, and the runtime must serialize the same fields in the
same order with the same caps. Keep this file and `CANONICAL` in sync with
dsh-totem; `tests/test_normalize_state.py` asserts the contract.

Canonical order: task, memory, changes, answer, toolCalls, grounding.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from typing import Any

CANONICAL = {
    "task": 400,
    "input": 4000,
    "output": 2000,
    "diff": 4000,
    "diff_total": 16_000,
    "answer": 200,
    "tool_calls": 10,
    "memory": 3,
    "memory_title": 48,
    "memory_statement": 120,
    "grounding": 2,
    "grounding_output": 240,
}


def clip(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= max_chars else text[:max_chars] + "…[truncated]"


def _split_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def unified_diff(change: dict[str, Any]) -> str | None:
    """Convert the edit tool's stored diff JSON into unified diff text."""
    raw = change.get("diff")
    if not raw:
        return None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return str(raw)
    if not isinstance(parsed, list) or not parsed:
        return str(raw)
    path = change.get("path") or ""
    header = f"--- a/{path}\n+++ b/{path}\n" if path else ""
    chunks: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        old = item.get("oldText") or ""
        new = item.get("newText") or ""
        for line in difflib.unified_diff(
            _split_lines(old), _split_lines(new), fromfile="a", tofile="b", lineterm=""
        ):
            if line.startswith(("---", "+++", "@@")):
                continue
            chunks.append(line)
    text = "\n".join(chunks)
    return header + text if text else None


def _budget_diffs(changes: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    ranked = []
    for index, change in enumerate(changes):
        if "diff" not in change:
            continue
        base = (change.get("path") or "").split("/")[-1].lower()
        mentioned = 1000 if len(base) > 2 and base in task else 0
        ranked.append((mentioned + index, index, change))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)

    kept: dict[int, str] = {}
    remaining = CANONICAL["diff_total"]
    for _, index, change in ranked:
        if remaining <= 0:
            break
        diff = clip(change.get("diff"), min(CANONICAL["diff"], remaining))
        remaining -= len(diff)
        kept[index] = diff
    return [
        change if index not in kept else {**change, "diff": kept[index]}
        for index, change in enumerate(changes)
    ]


def normalize_memory(memory: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for item in (memory or [])[: CANONICAL["memory"]]:
        row: dict[str, Any] = {}
        if item.get("id") is not None:
            row["id"] = str(item["id"])
        if item.get("type") is not None:
            row["type"] = str(item["type"])
        if item.get("title") is not None:
            row["title"] = clip(item["title"], CANONICAL["memory_title"])
        if item.get("statement") is not None:
            row["statement"] = clip(item["statement"], CANONICAL["memory_statement"])
        out.append(row)
    return out


def normalize_state(
    task: str,
    answer: str = "",
    changes: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    grounding: list[dict[str, Any]] | None = None,
    memory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the canonical state dict in the runtime key order."""
    clipped_task = clip(task, CANONICAL["task"])
    canonical_changes = []
    for change in changes or []:
        diff = unified_diff(change) if "diff" in change else None
        row: dict[str, Any] = {
            "tool": str(change.get("tool") or "apply_patch"),
            "input": clip(change.get("input"), CANONICAL["input"]),
            "output": clip(change.get("output"), CANONICAL["output"]),
        }
        if change.get("path"):
            row["path"] = str(change["path"])
        if diff:
            row["diff"] = clip(diff, CANONICAL["diff"])
        canonical_changes.append(row)

    calls = []
    for call in (tool_calls or [])[-CANONICAL["tool_calls"] :]:
        row = {"tool": str(call.get("tool") or call.get("name") or ""), "status": "completed"}
        args = call.get("title") or call.get("arguments")
        if args:
            row["title"] = clip(args, 240)
        calls.append(row)

    ground = []
    for item in (grounding or [])[-CANONICAL["grounding"] :]:
        row: dict[str, Any] = {"tool": str(item.get("tool") or "")}
        title = item.get("title") or item.get("input")
        if title:
            row["title"] = clip(title, 120)
        row["output"] = clip(item.get("output"), CANONICAL["grounding_output"])
        ground.append(row)

    return {
        "task": clipped_task,
        "memory": normalize_memory(memory),
        "changes": _budget_diffs(canonical_changes, clipped_task),
        "answer": clip(answer, CANONICAL["answer"]),
        "toolCalls": calls,
        "grounding": ground,
    }


def task_split(task: str, dev: int = 5, test: int = 5) -> str:
    bucket = int(hashlib.sha1(task.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 100
    if bucket < dev:
        return "dev"
    if bucket < dev + test:
        return "test"
    return "train"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Normalize decision JSONL states to the dsh-totem canonical shape.")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    with open(args.input) as source, open(args.output, "w") as sink:
        for line in source:
            row = json.loads(line)
            row["state"] = normalize_state(**row["state"])
            sink.write(json.dumps(row) + "\n")
    print(f"normalized {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
