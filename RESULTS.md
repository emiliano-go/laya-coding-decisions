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

## v6 run (2026-11-14, in progress)

First run on the execution-verified dataset (`scripts/gen_dataset_v6.py`).

- Data: `train_m6s.jsonl` = 60k sampled base rows + 36k new rows (SWE-bench-extra,
  R2E-Gym-Lite, CodeJudgeBench, perturbations, cross-instance). 288,498
  sequences; noul 156k / score 96k / choice 36k. Canonical state layout.
- Training: head-only, 1024/256, bf16, 8-bit AdamW, micro-batch 2, grad-accum 8,
  label smoothing 0.05, from `out-multi`; 1 epoch (18,031 updates).
- Completed: 15,062.9 s (4.18 h), 19.2 seq/s, avg loss 0.6992, peak 2.16 GB.
  Checkpoint: `/data/laya-ft/out-m6-head`.
- Gates: judge dev + `bug_class` accuracy, then the 33-case funnel and the
  `hard_ood_m6` adversarial slice. Deploy only if it beats `out-multi` on
  catch/false-pass with no false-alarm regression.
- Dev (`dev_m6`, n=7,599): `requirements_met` acc 0.771 / bal 0.765 / AUC
  0.849 / ECE 0.294; `bug_risk` exact 0.178 / MAE 1.173 / ECE 0.228;
  `bug_class` acc 0.654 (majority `none` = 0.650) / ECE 0.108;
  `no_unrelated_changes` bal 0.501 / AUC 0.801.
- OOD (`hard_ood_m6`, n=512): `requirements_met` bal 0.541 / AUC 0.562;
  `bug_risk` exact 0.004 / MAE 1.458; `bug_class` acc 0.502.
- Funnel (33-case, pass 0.65, veto 0.5):
  | model | catch | false pass | false alarm |
  |---|---|---|---|
  | out-multi | 21/22 | 1/22 | 6/11 |
  | out-m5mem4-head | **22/22** | **0/22** | **4/11** |
  | out-m6-head | 20/22 | 2/22 | 3/11 |
  Missed slop: unrelated-changes, refactor-beyond-request. False alarms:
  minimal-fix-good, answer-grounded-good, answer-debug-good.

### v6 verdict (2026-11-15)

**Do not deploy.** m6 improves dev `requirements_met` AUC and `bug_risk`
exact over base, and lowers false alarms (3 vs 4/6), but it loses one catch and
adds one false pass against the current `out-m5mem4-head` (20/22, 2 FP vs
22/22, 0 FP). `bug_class` is at the majority baseline, and `bug_risk` is still
near-constant. Keep `out-m5mem4-head` as production.

Next levers: hard-negative mining for the two missed scope cases, balanced
`bug_class` training, and an ordinal/calibrated `bug_risk` head. Starting the
next run from `out-m5mem4-head` (instead of `out-multi`) is the cheapest way to
protect catch while adding execution-verified data.

## v7 labels run (2026-11-15, in progress)

`train_v7` (118,496 rows) starts from **out-m5mem4-head** (protect catch),
head-only, 1024/256, 1 epoch, lr-head 5e-5, label smoothing 0.05, with
inverse class weights for `bug_class` (capped 10x). Four questions now include
`no_unrelated_changes`; teacher pairs provide balanced class labels; teacher
rows keep graded `bug_risk` 1/2/3; scope negatives are explicit.

Gates: dev `bug_class` balanced accuracy/macro, `no_unrelated_changes` balanced
accuracy/AUC, `bug_risk` exact/MAE, then the 33-case funnel with the four-question
set (must beat m5mem4's 22/22, 0 FP, 4 FA).

## Weight-Space Programming + calibration spike (2026-11-15)

Read-only diagnostic on the production `out-m5mem4-head`: the final scorer's
hidden representation separates `bug_risk` 0 vs 3 at **AUC 0.776** (balanced
200-row dev sample, difference-in-means) and the scorer's `p(level3)` already
has **AUC 0.773**. The expected score is compressed (means 1.39 / 1.96), so
`bug_risk` exact is poor even though the signal is present. This is the
"obstacle hiding an ability" case, so WSP was tested directly.

### WSP edits (`scripts/wsp_probe.py`)

A single rank-1 edit to `scorer.3.weight` along a paired direction
(`mean(h3 - h0)` or `mean(h | label3) - mean(h | label0)`), sweeping alpha.

| target | best edit | dev probe | verdict |
|---|---|---|---|
| `bug_risk` | state3 direction, α=64 | exact 0.064 → **0.502**, MAE 1.34 → **1.13**, AUC 0.690 → 0.690 | bias shift, not capability |
| `bug_class` | wrong-comparison − none, α=−64 | acc 0.396 → **0.421**, macro 0.122 → 0.125 | no signal |

Funnel for the `bug_risk` edit: **19/22 caught, 3 false pass, 3 false alarm**
(worse than m5mem4's 22/22, 0 FP, 4 FA). The edit mostly shifts the expected
score toward level 3; discrimination (AUC) does not improve.

### Calibration / decision-rule edits

| rule | caught | false pass | false alarm |
|---|---|---|---|
| m5mem4, expected score (production) | **22/22** | **0/22** | **4/11** |
| m5mem4, `bug_risk = p(level3)` | 19/22 | 3/22 | 3/11 |
| m5mem4, `noul` temperature 0.5 | 19/22 | 3/22 | 4/11 |
| m5mem4, `bug_risk` temperature 1.5 | 22/22 | 0/22 | 5/11 |

### Follow-up: per-option bias and head-only merges

**Per-option bias/prior (m5mem4), dev_v7:**

| target | baseline | best bias | result |
|---|---|---|---|
| `bug_class` macro | 0.145 (acc 0.290) | `+0.25·log prior` | macro **0.161**, acc 0.459 |
| `bug_risk` exact | 0.118 (MAE 1.244) | `+4·log prior` | exact **0.179**, MAE 1.198, p3 0.681 |

The class signal on m5mem4 is too weak for a prior fix; this is the classifier
example's lesson without the classifier's clean bias.

**Head-only merges with m6** (funnel, 3-question set):

| merge | caught | false pass | false alarm |
|---|---|---|---|
| m5mem4 (baseline) | **22/22** | **0/22** | **4/11** |
| task, α0.1 | 21/22 | 1/22 | 7/11 |
| task, α0.2/0.3 | 19/22 | 3/22 | 6–7/11 |
| task, α0.4 | 18/22 | 4/22 | 6/11 |
| interp, α0.2/0.3 | 22/22 | 0/22 | 8/11, 7/11 |
| scorer-only, α0.2/0.5 | 22/22 | 0/22 | 5/11 |

No merge improves the operating point. The m6 task vector is destructive;
interpolation preserves catch but adds false alarms. `v7` (trained from
m5mem4) remains the only promising merge candidate, and its training was still
running.

### Verdict

No WSP, calibration, bias, or m6 merge beats the production rule on the funnel. The
information is present (AUC ~0.69–0.78) but the current expected-score rule is
already the best catch/false-pass tradeoff; shifting it buys dev exact at the
cost of real catches. `bug_class` is label-limited (65% `none`), and a head
direction cannot add the missing classes. This matches the WSP blog's Gemma
counterexample: weight edits remove obstacles, they do not add abilities.

Keep `out-m5mem4-head`. If WSP is revisited, the next honest test is a linear
probe direction on a larger balanced set with a held-out test split, not the
33-case bench. The experimental bench flags live in
`packages/encoder/script/decision-bench.ts` (`DECISION_BUG_RISK_P3`,
`DECISION_NOUL_TEMPERATURE`, `DECISION_BUG_RISK_TEMPERATURE`).

## v7 labels run (2026-11-15, in progress)

`scripts/gen_dataset_v7.py` rebuilds the labels around the two failures from v6:

- four-question set: `requirements_met`, `no_unrelated_changes`, `bug_risk`,
  `bug_class` (scope gets its own head);
- teacher rows keep graded `bug_risk` 1/2/3; scope rows get
  `bug_class=scope-unrelated`;
- teacher preference pairs supply balanced `bug_class` labels from
  `dimensions.defect` (wrong-comparison, off-by-one, null-empty, type-coercion,
  resource-leak, swallowed-exception, other);
- deterministic class mutations and 3,000 synthetic scope negatives;
- class-weighted `bug_class` loss (weights clamped at 10x);
- split grouped by task hash; train/dev/test: 118,496 / 8,749 / 8,816.
- dev `bug_class`: none 1,058, other 210, scope 197, wrong-comparison 169,
  null-empty 72, off-by-one 66, type-coercion 86, swallowed 46, resource 3.

Training: head-only from `out-m5mem4-head`, 1024, lr-head 5e-5, label smoothing
0.05, class balance, 1 epoch. Completed in 16,698.6 s, avg loss 0.7748.
Checkpoint `/data/laya-ft/out-v7-head`.

**v7 results (do not deploy):**

| metric | m5mem4 | v7 |
|---|---|---|
| `bug_class` macro (dev) | 0.145 | **0.228** |
| `bug_class` acc (dev / OOD) | 0.290 / 0.502 | 0.371 / 0.178 |
| `bug_risk` exact / MAE (dev) | 0.118 / 1.244 | 0.077 / 1.259 |
| `bug_risk` exact (OOD) | 0.004 | 0.002 |
| `no_unrelated_changes` bal / AUC | — | **0.508 / 0.646** |
| `requirements_met` bal / AUC (dev) | ~0.765 / 0.849 | 0.632 / 0.742 |
| `requirements_met` bal / AUC (OOD) | 0.541 / 0.562 | 0.502 / 0.524 |
| 4-question funnel | 19/22, 3 FP, 4 FA | 19/22, 3 FP, 5 FA |
| 3-question funnel | **22/22, 0 FP, 4 FA** | 21/22, 1 FP, 5 FA |

The scope head is at chance because the labels are 88% "unrelated=true"; the
model predicts true. Adding the fourth question to the runtime also drops
m5mem4 from 22/22 to 19/22, 3 FP (the extra weight dilutes bug catch), so the
question set was reverted to three.

### Deterministic scope rule (kept)

A high-precision rule in `verify.ts` flags repo-wide formatters
(`prettier/eslint/ruff/black/... --write|--fix .`) and bulk mechanical rewrites
(`rewrote N lines`, `updated N call sites`, `across N files`). On the 3-question
funnel with m5mem4: **22/22 caught, 0 false pass, 4 false alarm**, deterministic
catches 15 → 17 (the two scope cases no longer depend on the judge). This is the
scope fix; the model head was not usable.


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

## Round 3 - runtime gate hardening (2026-11-15)

The judge is now review-only where it cannot steer safely. Turn-level
`needs_review` never blocks or retries; only deterministic high-severity
findings do. Per-tool Laya calls in encoder and dsh are telemetry, not gates.
`reviewBand` resolves marginal vetoes and near-threshold composites to
`needs_review(reviewOnly=true)`, and `confidenceFloor` does the same for
low-confidence answers.

Hard failures stay hard: a severe veto or a composite below
`passThreshold - reviewBand` is `fail` even when confidence is low. Without
this, a failing baseline (typeguard) was labelled review-only because its
answers were uncertain.

Visible5 rerun (`confidenceFloor=0.5`, production checkpoint):

| task | off | veto-turn | terminal |
|---|---|---|---|
| yamllint | pass | pass | review-only pass 0.696 |
| exceptiongroup | pass | pass | review-only pass 0.628 |
| typeguard | fail | fail | hard fail 0.362 |
| sqlparse | pass | pass | review-only pass 0.672 |
| bottle | pass | pass | review-only pass 0.660 |
| boltons | pass | pass | review-only pass 0.706 |

5/6 match baseline, no false alarms, no timeouts (max 317s). A targeted
typeguard rerun after the hard-fail fix terminated at 363s with terminal
`fail` and no review-only flag.

A 20-task hard-set A/B is queued after baseline validation; the first
`tasks.json` batch was mostly baseline-passing (invalid) instances, so the
candidate set is now baseline-filtered before the A/B.

Follow-up: a clear pass (`composite >= passThreshold + reviewBand`) is never
downgraded by low confidence. The dsh profile floor is now 0.2; a 0.807 clear
pass passes, while a 0.654 margin (+0.004, inside the 0.05 band) remains
review-only by the band rule, not by confidence. The 0.654 case has
`bug_risk` confidence 0.047, so lowering the floor to 0.2 would not have
changed it. Set `layaReviewBand: 0` to make marginal passes gate.

## Localization

Zero-shot hunk ranking (`scripts/localize_hunks.py`, 100 mutation rows) scores
top1 **0.30**, top2 **0.68**, MRR **0.578**; function-level localization is
weaker (recall 0.333, top5 0.5). The hunk step is the usable narrowing signal;
more retrieval work is needed before it can replace a human review step.

`scripts/build_localization.py` previously wrote rows without per-row
`criteria` and with numeric gold keys (`0/1/2`), so `train_coding.py` skipped
them. Fixed to emit `hunk-*` criteria and matching gold keys; regenerated
`loc_{train,dev,test}.jsonl` (1,653 / 220 / 80). No localization fine-tune has
been run yet.

Runtime test path: dsh-totem `POST /totem/laya/localize` (manual, non-gating)
accepts `{task, diff}` or `{task, changes}` and returns ranked hunks; verified
on a two-hunk diff (top 0.5294 vs 0.4706, 254 ms) while the automatic gate was
off.

### A/B: 10 baseline-valid tasks

| condition | pass | mean s | mean steps | review-only | flips |
|---|---:|---:|---:|---:|---:|
| off | 8/10 | 139.3 | 0 | 0 | - |
| veto-turn | 8/10 | 201.8 | 27.8 | 3 | 0 |
| veto-tool | 8/10 | 182.2 | 27.8 | 2 | 0 |

Failures were hydra and typeguard in every arm; no arm fixed either. The gate
bought no success and added ~40-60 s per task. Typeguard's veto-turn arm hit
the 10-minute harness timeout (603 s vs 95.9 s off); its veto-tool arm took
460 s. No false alarms and no verdict flips. The 20-task candidate file was
mostly baseline-passing (invalid) instances; only 10 unique tasks were valid
after baseline filtering.

### Localization head-only fine-tune

Built `out-loc-head` from `out-m5mem4-head` (head-only, 26.5M trainable, 3
epochs, 1,653 rows, avg loss 1.15). On the same 100-row hunk probe: top1
**0.28**, top2 **0.68**, MRR **0.572** vs zero-shot 0.30 / 0.68 / 0.578. No
improvement; reject. Keep zero-shot/manual `POST /totem/laya/localize` and
treat localization as an aid, not a gate.

### Confirmation: remaining baseline-valid tasks

| condition | pass | mean s | mean steps | flips |
|---|---:|---:|---:|---:|
| off | 9/11 | 129.1 | 0 | - |
| veto-turn | 9/11 | 166.7 | 18.2 | 0 |

Combined with the first A/B: **21 valid Python tasks, 17/21 pass in both
arms, zero flips, zero false alarms.** Failures were identical (hydra,
typeguard, cantools, dask). The gate added turns and latency but fixed
nothing. Three Go repos plus goatcounter were skipped as unsupported; the
25-task valid pool therefore contains 21 runnable tasks.

**Decision:** stop Laya gate research. Keep the judge advisory/review-only
(dsh Laya off by default), keep localization as the zero-shot manual
narrowing endpoint, and do not train further heads on the current labels.
The remaining work is product/harness value, not gate tuning.

### Memory warm-start A/B (2026-11-16)

Seeded three non-oracle repo facts per memory arm via Totem (repo type, test
command, top-level layout); the agent runs with a memory-first prompt hint and
the judge is off in both arms.

| condition | pass | mean s | memory calls | flips |
|---|---:|---:|---:|---:|
| off | 8/10 | 163.1 | 0 | - |
| memory | 8/10 | 185.7 | 0 | 0 |

The seed was readable by encoder `digest()` (3 memories), but the agent made no
memory tool calls and the arm was slower on 7/10 tasks. No success or step gain;
reject seeded runtime facts as a harness lever. Memory is better used as a
durable run ledger / recall surface than as a warm-start hint.

## Failure-driven localization (2026-11-16)

Asked Laya where a bug is *after a test failure*: repo-wide function candidates,
failing pytest output as the state, gold = the function containing the injected
SWE-smith bug. 21 Python instances, 34 gold sites, zero-shot, no training.
Full record: `docs/failure-localization.md`.

| config | top1 | top3 | MRR |
|---|---:|---:|---:|
| Laya, task + failure | 0.235 | 0.471 | 0.390 |
| H3 deterministic shortlist order | 0.176 | 0.412 | 0.340 |
| H1 innermost source frame | 0.176 | 0.412 | 0.340 |
| H2 BM25 only | 0.147 | 0.382 | 0.306 |
| task text only (old issue probe) | 0.147 | 0.294 | 0.284 |

Laya is consistently ahead but the paired bootstrap CI for `L − H3` on MRR is
[−0.023, 0.185]: not separated from zero at n=21. In the assertion-only stratum,
where the traceback gives nothing, Laya ties the deterministic order exactly
(0.214 vs 0.214).

Retrieval is the binding constraint, not ranking: R@12 is 0.647 (`k=16`: 0.676).
Eight of the twelve missed gold sites are in files that never appear in any
traceback frame — a pytest failure carries no signal for them. Conditional on the
answer being in the shortlist, Laya wins 3 instances, loses 1, ties 17 vs H3
(sign test p=0.31).

**Decision:** the retrieval gate (R@12 ≥ 0.85) fails, so the route/tool/UI work
was not built. This is the fourth independent negative for Laya as a signal source
(gate A/B 17/21 flat, loc head-only fine-tune flat, memory warm start flat,
failure localization not separable). Keep `POST /totem/laya/localize` as the
manual diff-narrowing aid; keep the gate off by default. Do not invest further in
autonomous or semi-autonomous localization.

### Call-shape follow-up (2026-11-16)

Tested whether asking differently recovers signal: short labels, code snippets as
options, per-candidate `noul` scoring, pairwise + Bradley-Terry, and the other two
checkpoints. Nothing beats the original 12-way choice (top1 0.235 / MRR 0.385 vs
deterministic 0.176 / 0.340). Short labels are identical (0.235), pointwise is
0.088, pairwise at k=8 is 0.147 against a 0.176 control, `typed-decisions` ties
english, `multilingual` collapses to 0.059. The shared-option-budget hypothesis
is falsified: the model was not being starved of evidence. What remains is
retrieval (coverage / import graph) and more instances, neither of which is a
Laya change. Details in `docs/failure-localization.md`.

### Hierarchical narrowing (2026-11-16)

The dir → file → section → function cascade was never implemented; it existed
only as a proposal. Implemented and tested against the same corpus: pure cascade
top1 **0.000** / MRR 0.128, hybrid (cascade to 3 files, then in-file ranking)
top1 0.029 / MRR 0.112, both far below the deterministic flat shortlist
(0.176 / 0.340) and the single 12-way choice (0.235 / 0.385). File-level top1 is
0.190 and the in-file ranking only wins 1 in 4 when the file is right; the
cascade gravitates to build scripts and tooling modules named in the failure
text. Reject the hierarchy: the coarse stage must be a better retriever than
BM25, and Laya is not one.

### N-best patch selection (2026-11-16)

Tested the last shape where a "catches bugs" capability would change an outcome:
choosing among candidate patches. SWE-Rebench was already on disk (6,291 tasks,
4,808 with 4-12 execution-labelled candidates, base rate ~0.46). Laya's pick is
at chance in every setup — 0.465 vs 0.482 over 600 tasks, 0.657 vs 0.665 over
201 four-candidate tasks with richer options, 0.633 vs 0.654 pointwise. The
probability mass on correct candidates equals the base rate, i.e. a uniform
distribution over the candidates. Also worth recording: the old funnel's 12/14
catch came from `requirements_met` plus the deterministic verify lane, not from
`bug_risk`; a binary requirement check cannot order candidates, which is what
these uniform distributions demonstrate.
