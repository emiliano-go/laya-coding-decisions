# Memory triage with Laya — plan

Status: **planned**, not started. 2026-11-16.
Prerequisite reading: [`CONCLUSIONS.md`](CONCLUSIONS.md) — the rule this plan
follows is that a Laya experiment needs a closed label set, a labeled corpus,
and a baseline it must beat *before* it runs.

## 0. Why this shape

These are the only tasks in the stack that fit what Laya actually is: text in,
one label out, small closed set, not a comparison between two code artifacts.
The catch is that Totem already ships embeddings (`minishlab/potion-retrieval-32M`),
so the honest baseline is a nearest-neighbour classifier, not a regex.

Both tasks are **batch/offline**. Neither goes back into the agent loop, even if
it passes.

## Task 1 — memory type classification

**Question:** given a memory's title and statement, can Laya assign its type
better than a cheap embedding kNN?

**Product decision it would change:** type a candidate memory automatically
instead of asking the human, and flag memories whose recorded type is wrong.
Value is modest — the type is metadata, not a gate.

**Labels (measured 2026-11-16):** 490 memories across 10 project DBs.

| type | n |
|---|---:|
| implementation | 336 |
| gotcha | 48 |
| decision | 45 |
| architecture | 39 |
| observation | 8 |
| bug | 6 |
| invariant | 5 |
| contract | 3 |

Majority accuracy is 0.686, so **accuracy is the wrong metric**. Use macro-F1;
the majority classifier scores ≈ 0.11 macro-F1. Eight classes with n=490 is
small — report a bootstrap CI and per-class support.

**Inputs:** `title` + `statement` (≤ 400 + 2000 chars, the canonical caps).

**Leakage rules (all mandatory):**

- strip tags that name a type (`gotcha`, `decision`, `implementation`, …), and
  strip the `type` word if it appears in the title;
- group by project: evaluate leave-one-project-out so project vocabulary cannot
  carry the label;
- no `metadata.memoryClass`, no `assertedBy`, no id prefixes.

**Baselines:**

1. majority class (0.686 accuracy / 0.11 macro-F1);
2. keyword rules over the statement (e.g. "do not", "must", "gotcha", "instead");
3. **embedding kNN** (k = 5, cosine) using Totem's own embedding backend —
   this is the bar.

**Metric and bar:** macro-F1 over the 8 classes, leave-one-project-out.

> Pass = Laya ≥ kNN + 0.10 macro-F1 **and** ≥ 0.55 absolute.

**Effort:** ~2 h. `scripts/eval_memory_type.py`: reads DBs through the `totem
export` CLI, builds rows, runs the three baselines plus one Laya `choice`
question, writes a per-class table.

## Task 2 — contradiction detection

**Question:** given two memories from the same project, are they contradictory?
Binary, closed, text-only.

**Product decision it would change:** surface real contradictions in the
conflicts inbox without an LLM pass, and stop asking the model on every write.

**Status: not evaluable today.** There are **4** recorded conflict resolutions
across all 10 projects, and **1** memory candidate. That is not a dataset.

Two ways forward, in order of honesty:

1. **Owner-adjudicated set (recommended).** Build ~100 pairs from same-project
   nearest neighbours plus pairs that share a subject/file/entity; the owner
   labels 50 positives / 50 negatives (~45 min). Then evaluate.
2. **Stronger-model labels.** Have the agent label the pairs and evaluate Laya
   against it. This measures *agreement with a stronger model*, not truth, and
   must be reported as such.

**Baselines:** embedding cosine threshold (the mechanism Totem already uses for
consolidation and duplicate detection), and a shared-tags heuristic.

**Metric and bar:** precision/recall at a chosen threshold.

> Pass = F1 ≥ cosine-threshold F1 + 0.15 with precision ≥ 0.80.

**Effort:** ~1 h to build the pairs + 45 min of labeling + ~1 h to evaluate.
**Do not build the classifier before the labels exist.**

## Order and budget

| step | time | gate |
|---|---|---|
| Task 1 harness + run | 2:00 | pass/fail against kNN, recorded either way |
| Task 2 pair construction | 1:00 | pairs on disk, ready for labeling |
| Task 2 labeling | 0:45 (owner) | 100 adjudicated pairs |
| Task 2 evaluation | 1:00 | pass/fail against cosine |

Task 1 can start immediately; Task 2 is blocked on labels.

## What happens on a pass

- Task 1: a `totem type` *suggestion* in the candidates UI. Advisory, never
  auto-written, never in the agent loop.
- Task 2: pairs appear in the existing conflicts inbox; the owner still
  resolves them.

## Risks

- **Label noise.** The types were chosen by agents over months; some are wrong.
  The eval measures agreement with past choices, not truth. Say so in the
  results.
- **Tag leakage** is easy to introduce and would inflate every number.
- **Small n.** 490 items, 8 classes: a 0.10 macro-F1 delta is at the edge of
  what this corpus can resolve; report the CI and refuse to over-read it.
- **Embedding baseline is strong.** If Laya cannot beat a 32M embedding kNN,
  the honest answer is to use the kNN.
