# Conclusions — the Laya coding-judge program (2026-09 → 2026-11)

**Verdict: Laya has no measured role inside a coding agent, either as a gate or
as an assistant. The line is closed.** The one capability that survived every
round is a deterministic verify lane plus a binary `requirements_met`
judgement, and the latter cannot rank candidates, which is what an agent
actually needs.

This document is the single readable summary. `RESULTS.md` holds the raw rounds
and `docs/failure-localization*.md` holds the last two experiments in full.

## The question

Can a small (≈300M parameter), non-autoregressive, ~33 ms decision model make a
coding agent better — by gating it, warning it, narrowing where a bug is, or
choosing between candidate patches?

## What was tested

| round | task | n | result | verdict |
|---|---|---|---:|---|
| 1 | Turn gate, veto on/off | 21 valid Python tasks | 17/21 pass in **both** arms, zero flips, zero false alarms, +40–60 s/task | reject |
| 2 | Head-only fine-tune (`out-loc-head`, v6/v7 heads) | 1,653 rows | dev `bug_risk` exact 0.118 → 0.179; funnel 10/14 vs base 12/14 | reject |
| 3 | Memory warm start (3 seeded repo facts) | 10 tasks | 8/10 both arms, slower on 7/10, **0** memory tool calls | reject |
| 4 | Hunk localization (rank hunks of a diff) | 100 mutation rows | top1 0.30 / top2 0.68 / MRR 0.578; head fine-tune 0.28 / 0.68 / 0.572 | manual only |
| 5 | Failure-driven localization (rank repo functions after a pytest failure) | 21 instances / 34 sites | R@12 **0.647**; Laya 0.235 top1 vs deterministic 0.176, CI [−0.023, 0.185] | reject |
| 6 | Call shapes (short labels, pointwise, pairwise, 3 checkpoints) | same 21 | short 0.235 = long; pointwise 0.088; pairwise 0.147 vs 0.176; `typed-decisions` ties; `multilingual` 0.059 | reject |
| 7 | Hierarchical narrowing (dir → file → class → function → window) | same 21 | pure cascade top1 **0.000** / MRR 0.128; hybrid 0.029 / 0.112; file top1 0.190 | reject |
| 8 | N-best patch selection (SWE-Rebench, execution-labelled) | 600 + 201 tasks | pick@1 0.465 vs 0.482 random; 0.657 vs 0.665 with richer options | reject |

## Findings that generalize

1. **A binary judgement cannot rank.** `requirements_met` is real (dev AUC 0.82)
   and is what drove the old 12/14 funnel catch — together with the
   deterministic verify lane, not `bug_risk`. But "does this patch satisfy the
   request" is one yes/no per patch; it produces a *uniform* distribution when
   asked to choose between patches (round 8: correct-candidate mass equals the
   base rate in every setup).
2. **Retrieval is the wall, and it is not a model problem.** After a test
   failure, 8 of 12 missed bug locations live in files that never appear in any
   traceback frame. Neither BM25 nor the model can reach them; a coverage or
   import-graph signal could.
3. **Cheapness does not create signal.** 12 calls per instance (pointwise),
   28–66 pairwise comparisons, or a 5-level beam cascade all scored at or below
   a single well-formed 12-way choice. More inference bought nothing.
4. **Prompt shape was not the constraint either.** The choice head shares a
   192-token option budget; giving options short labels (freeing the budget)
   changed the result by 0.000 top1.
5. **The deterministic lane won every head-to-head.** Traceback frames + BM25
   beat every Laya variant on localization; verify rules drove the funnel catch.
   A 30-line heuristic is the honest baseline for this class of decision.

## What is worth keeping

- **The corpus and the harness.** 21 materialized SWE-smith instances with
  captured failures, gold spans, frozen candidate extraction, and a fixed
  evaluation bar (`scripts/build_failure_loc.py`, `scripts/localize_*.py`). Any
  future retrieval idea can be measured against it in ~10 minutes.
- **The negative record.** Rounds 1–8 mean nobody has to re-litigate this.
- **`requirements_met` as an advisory review signal**, if a human-facing review
  queue ever wants a cheap prior. Never as a gate.

## What not to do next

Do not retrain heads on the current labels, do not add gate arms, do not
re-tune prompts, do not try more call shapes, and do not expect a small
classifier to choose between patches. If a selector is wanted, it must read
code.

## If the question is reopened

The bar is set: a labeled task, a stated threshold **before** the run, and a
deterministic baseline to beat. The only untried lever is retrieval
(coverage/import graph), and it is not a Laya change.

## Integration status

On 2026-11-16 the Laya integration was removed from `dsh-totem`: no managed
server, no turn gate, no advisory calls, no `laya_localize`, no Laya tab or
cards. What remains there is the deterministic verify lane, which is what
actually produced the catch rate this program was chasing.
