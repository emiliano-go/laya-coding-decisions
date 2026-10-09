# Results

Judge-dev on the corrected `dev_m5d` (tuned 2-question set), and the offline
funnel bench (`encoder` `decision-bench.ts`, tuned set):

| checkpoint | bug_risk exact | bug_risk MAE | req bal | req AUC | funnel catch | false pass |
|---|---|---|---|---|---|---|
| `out-multi` (base `convaiinnovations/laya`) | 0.011 | 1.233 | 0.783 | 0.823 | **12/14** | **2/14** |
| `out-m5b` (v3, coupled labels) | 0.571 | 1.217 | 0.761 | 0.830 | — | — |
| `out-m5c` (v4, decoupled labels) | 0.469 | 1.123 | 0.760 | 0.830 | 10/14 | 4/14 |

## Findings

1. **Full fine-tuning from the base checkpoint degrades the funnel.** Every run
   (`out-m5`, `out-m5b`, `out-m5c`) lost catch and gained false passes, and
   forgot `requirements_met` / `decision_correct`, even when `bug_risk` *exact*
   improved. This is catastrophic forgetting of the reasoning heads, not
   undertraining — epoch 2 (stacking on a degraded model) does not fix it.
2. **Decoupling the labels (v4) did not recover the funnel.** It tripled
   `bug_risk` exact (0.011 → 0.469) but catch still fell (12 → 10) and
   `requirements_met` stayed below the base. The base checkpoint remains the best
   judge.
3. **Head-only fine-tuning is the next experiment** (`--trainable head`, encoder
   frozen): preserve the base head reasoning while adapting `bug_risk`.

## Notes

- `bug_risk` exact ≈ 0 for the base checkpoint: it emits a near-constant level,
  yet the funnel still catches 12/14 via `requirements_met` + the deterministic
  verify lane. Bench catch is **not** driven by `bug_risk` exact.
- Bench control reproduces the documented baseline exactly (12/14, FP 2/14) on
  the current evidence-less wire.
