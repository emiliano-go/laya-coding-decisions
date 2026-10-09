# Results

Every run below starts from the base `convaiinnovations/laya` ("out-multi") and
is judged on `dev_m5d` (corrected labels) and the offline funnel bench
(`encoder` `decision-bench.ts`, tuned 2-question set). The base is the control.

| checkpoint | setup | bug_risk exact | req bal | req AUC | decision_correct | funnel catch | false pass | false alarm |
|---|---|---|---|---|---|---|---|---|
| out-multi | base | 0.011 | 0.783 | 0.823 | 0.898 | **12/14** | **2/14** | 3/5 |
| out-m5b | v3, full, coupled labels | 0.571 | 0.761 | 0.830 | 0.884 | n/a | n/a | n/a |
| out-m5c | v4, full | 0.469 | 0.760 | 0.830 | 0.883 | 10/14 | 4/14 | 2/5 |
| out-m5c-head | v4, head-only, 512 | 0.348 | 0.822 | 0.892 | 0.899 | 10/14 | 4/14 | 3/5 |
| out-m5c-head768 | v4, head-only, 768 | 0.350 | 0.826 | 0.894 | 0.898 | 9/14 | 5/14 | 3/5 |
| out-m5e-head768 | v5, head-only, 768, 3 questions | 0.359 | 0.808 | 0.880 | 0.898 | 10/14 | 4/14 | 3/5 |

## Findings

1. No training run beat the base checkpoint on the funnel. Full fine-tuning
   forgets the reasoning heads (catch 12 to 10, false pass 2 to 4). Head-only
   preserves them (decision_correct 0.899, requirements_met bal up to 0.826) but
   the OOD funnel still trails the base.
2. Longer context did not help. ModernBERT supports 8192 tokens, but raising the
   training window from 512 to 768 left the dev metrics unchanged and made the
   funnel slightly worse (9/14). The truncated diff was not the bottleneck.
3. Training `no_unrelated_changes` (v5, 3 questions) did not make it
   discriminative: balanced accuracy 0.500, and the funnel is identical with or
   without it in the tuned set. The head predicts positive almost always.
4. A decision-rule sweep on the base (E0, no training) had no clean win.
   `no_unrelated_changes` reaches 14/14 catch and 0 false pass, but only via a
   blunt veto that also flags all 5 good cases (false alarm 5/5); relaxing the
   veto gives up the catch instead.

## Why

The synthetic dataset does not cover the out-of-distribution funnel cases
(semverlite buggy-guard and no-backtracking, unrelated-changes, CF cases), and
the base model's broad pretraining generalizes to them better than anything the
synthetic labels teach. Improving dev metrics (requirements_met) did not move
the funnel, because the funnel is dominated by cases the data does not contain.

## Recommendation

Keep the base `convaiinnovations/laya` as the production judge. The next lever,
which this round did not build, is execution-verified labels (SWE-smith
bug-injection with unit-test verification, SWE-Gym task outcomes) plus a
preference objective and teacher distillation. Do not spend GPU budget on
synthetic-only retraining.

## Notes

- `bug_risk` exact is near 0 for the base: it emits a near-constant level, yet
  the funnel still catches 12/14 via `requirements_met` plus the deterministic
  verify lane. Bench catch is not driven by `bug_risk` exact.
- Head-only is about 3x faster than full (3.3h vs 8.9h at 512; 4.6h at 768 for
  three questions) and uses about 2 GB.
- Bench control reproduces the documented baseline exactly (12/14, false pass
  2/14) on the current evidence-less wire.
