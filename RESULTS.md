# Results

## 2026-11 update

### Corrected labels (major data bug)
The original `gold.jsonl` mixed SWE-Gym fix patches with **SWE-smith breaking
patches**: SWE-smith's `patch` field is the bug-*injection*, not the fix, so
~17.5k buggy patches were labelled `correct=True`. The cache is rebuilt as
`gold_fixed.jsonl` (SWE-Gym = correct, SWE-smith = a high-quality negative), and
`fetch_dec.py` now labels them correctly. `gen_dataset_v2.py` consumes the fixed
cache (`train_m5f`).

### Trained judges (all start from `out-multi` unless noted)
| checkpoint | setup | dev result | OOD funnel |
|---|---|---|---|
| out-m5f-head | corrected labels (m5f), head-only, 512 | bug_risk exact 0.008 -> **0.346**, MAE 1.46 -> **1.09**; req AUC 0.635 -> **0.830**; no_unrelated AUC 0.392 -> **0.778** | **10/14** caught (no transfer) |
| out-m5final-head | m5f + teacher bug/scope + answer-quality data | separation gap 0.06 -> **0.16**; `answer-good` fixed | 22/22, 0 FP, FA 5/11 |
| out-m5mem-head | memory-conditional (8x) | memory semantics correct, but lost 4 catches | 18/22, 4 FP |
| out-m5mem2-head | memory-conditional (3x) | memory semantics correct, lost 3 catches | 19/22, 3 FP |
| out-m5mem4-head | new layout (diff before answer), 1024, counterfactual memory | best deployable checkpoint | **22/22, 0 FP, FA 4/11** (pass 0.65) |

Conclusions: the label fix improves in-distribution discrimination but **does
not transfer OOD**; memory-conditional training fixes invariant semantics but
costs real catches; the model itself never beats the base on the OOD funnel.

### Model-based memory invariants: rejected
A dedicated `(invariant, patch) -> honors_invariant` stage was trained three
ways. Head-only on the full diff: ~0.41 for both honored and violating patches
(no separation). Full fine-tune: collapsed to a constant ~0.18. Hunk-delta
input: ~0.40 vs ~0.29, overlapping. The 421M encoder cannot learn invariant
consistency from these data. **Memory enforcement moved to Totem's runnable
`verificationMethod` (`run` + `appliesTo`), executed by the deterministic lane.**

### Deterministic lane is the reliable lever
Added, all tested: diff-aware scanning (secrets/deserialization were missed when
only the tool-input summary was scanned), assignment-in-condition, silent
swallow, unsafe deserialization, test-weakening, and answer-text safety. Also
runnable memory invariants (Totem `verificationMethod.run`, `appliesTo` globs)
and verification stamping on success. Offline bench (33 cases):

| judge + lane | caught | false pass | false alarm |
|---|---|---|---|
| out-multi (base) | 21/22 | 1/22 | 6/11 |
| out-m5final-head | 22/22 | 0/22 | 5/11 |
| out-m5mem4-head (pass 0.65) | **22/22** | **0/22** | **4/11** |

### Step-level checks (in progress)
`stepChecks: off | veto | full`, `stepCadence: turn | tool | both`; laya is
called per mutating turn/tool (~23 ms, unlimited), feedback is budgeted. On a
hard SWE-smith set (20 instances, DeepSeek v4.1 flash): the baseline passes
almost everything; step calls 39-48/run; 21-36% of step evaluations flag
intermediate states that later pass; `full` costs ~+47% runtime with 2/6
timeouts; **no outcome improvement measured yet**. The only apparent failure was
a harness bug (scoring a Go repo with `python`). Headroom requires harder tasks
or a weaker model.

## Earlier results (v1-v2)

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
