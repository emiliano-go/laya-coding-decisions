"""Runnable check for the coding-decisions dataset builder (no GPU, stdlib only).

    python recipes/coding-decisions/scripts/smoke_test.py

Asserts the two invariants that matter for the judge: memory context is derived
from the task alone (never the label), and `requirements_met` / `bug_risk` are
decoupled.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_dataset_v2 as g  # noqa: E402


def main() -> None:
    task = "fix the off-by-one error in the pagination loop in src/pager.ts"

    assert g.derive_memories(task) == g.derive_memories(task), "memory must be deterministic"

    right = g.make_row(task, "+ a\n- b\n", True)
    wrong = g.make_row(task, "+ a\n- b\n", False)
    assert right["state"]["memory"] == wrong["state"]["memory"], "memory must not depend on the label"
    assert list(right["state"].keys())[:2] == ["task", "memory"], "memory must lead the state"

    perturbed = g.make_row(task, "+ a\n- b\n", True, extra_gold={"bug_risk": g.score(3)})
    assert perturbed["gold"]["requirements_met"]["label"] == 1
    assert perturbed["gold"]["bug_risk"]["label"] == 3

    foreign = g.make_row(task, "+ a\n- b\n", False, extra_gold={"bug_risk": g.score(0)})
    assert foreign["gold"]["requirements_met"]["label"] == 0
    assert foreign["gold"]["bug_risk"]["label"] == 0

    empties = sum(1 for i in range(500) if not g.derive_memories(f"task {i}"))
    assert 50 < empties < 150, f"expected ~20% empty-memory rows, got {empties}/500"

    print("coding-decisions dataset smoke: OK")


if __name__ == "__main__":
    main()
