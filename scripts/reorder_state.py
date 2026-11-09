"""Rewrite state dicts to the serving key order: task, memory, changes, answer, ..."""
import json
import sys

ORDER = ["task", "memory", "changes", "answer", "toolCalls", "evidence", "grounding"]


def reorder(path):
    rows = 0
    with open(path) as src:
        data = [json.loads(line) for line in src if line.strip()]
    with open(path, "w") as out:
        for row in data:
            state = row.get("state")
            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except json.JSONDecodeError:
                    state = None
            if isinstance(state, dict):
                ordered = {key: state[key] for key in ORDER if key in state}
                for key, value in state.items():
                    if key not in ordered:
                        ordered[key] = value
                row["state"] = ordered
            out.write(json.dumps(row) + "\n")
            rows += 1
    print(f"{path}: {rows} rows reordered")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        reorder(path)
