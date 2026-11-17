# Memory type classification — implementation plan

Status: **ready to build**. 2026-11-16.
Parent: [`memory-triage-plan.md`](memory-triage-plan.md) (Task 1). This file is
the buildable breakdown: data, baselines, the Laya call, metrics, the bar.

## 0. What this decides

Given a memory's title and statement, can Laya pick its type better than the
existing embedding neighbours? One call per memory, offline, batch. If it
passes, the only honest product is an **audit**: a queue of memories whose
recorded type disagrees with the classifier. Never auto-typing, never in the
agent loop.

## 1. Data — `scripts/eval_memory_type.py collect`

```sh
python scripts/eval_memory_type.py collect --out .work/memory-types.jsonl
```

- discover every `*/.totem/totem.db` under a root (default
  `/home/eclipse/Documents/GitHub`), resolve the project dir, and read items
  through `totem --project <dir> export` (the libSQL files cannot be opened by
  stock sqlite3 — the FTS schema fails to parse).
- keep `{id, project, type, title, statement, tags, createdAt}`.
- drop anything not `status == "active"`.

Measured on 2026-11-16: **490 rows, 9 projects with items** (trustsight 145,
LibreCourseUY 118, dsh-totem 64, srcmem 61, totem 56, qwgpui 31, dbwarden-org 8,
aster-qt 5, test 2).

**Leakage scrub (mandatory, or the numbers are meaningless):**

- 59 rows (12%) carry their own type as a tag (`tags: [gotcha, …]`). Strip any
  tag whose value is a type name.
- 3 titles contain the type word. Flag them (`leak: title`) and report the
  headline metric both with and without them.
- never feed `metadata.memoryClass`, `assertedBy`, or the id.

**Classes:** the 8 that occur — `implementation` 336, `gotcha` 48, `decision`
45, `architecture` 39, `observation` 8, `bug` 6, `invariant` 5, `contract` 3.
Everything else has zero support and is excluded from the criteria.

## 2. Evaluation settings

| setting | neighbours | why |
|---|---|---|
| **in-project LOO** (headline) | same project only | how it would actually be used: a new memory against that project's history |
| cross-project | other projects | generalization check; also the only option for the 5-item projects |

Laya is zero-shot, so both settings score the same 490 predictions; only the
kNN baselines change.

## 3. Baselines

1. **B0 majority** — per setting (in-project majority, else global).
2. **B1 TF-IDF kNN** — stdlib only: tokenise `[A-Za-z_][A-Za-z0-9_]{2,}`,
   idf over the corpus, cosine, k=5, majority vote over neighbours.
3. **B2 embedding kNN** — `minishlab/potion-retrieval-32M` (the model Totem
   already configures) via `uv run --with model2vec --with numpy`, same k=5.
   If the package or the model cannot be fetched, record it and fall back to
   B1; the bar then applies to B1.

B2 is the real bar. If Laya cannot beat a 32M embedding kNN, the answer is to
use the kNN.

## 4. The Laya call

One `POST /v1/systemone` per memory:

```json
{
  "state": {"task": "<title, 200 chars>", "answer": "<statement, 1500 chars>"},
  "questions": {"type": {"type": "choice",
    "instructions": "Which type best describes this engineering memory?",
    "criteria": {"implementation": "what a piece of code does: API, function, schema",
                 "gotcha": "non-obvious behaviour or an error that bites later",
                 "decision": "a choice that was made, and why",
                 "architecture": "how a component or module is structured",
                 "observation": "a measured fact about the environment or runtime",
                 "bug": "a defect that was found",
                 "invariant": "a rule that must always hold",
                 "contract": "an interface guarantee between components"}}}
}
```

490 calls at ~0.9 s serial, ~8 min with 4 workers on CPU. Short option labels
are unnecessary here: type names are 1–2 tokens, so the 192-token option budget
is not a constraint at 8 options.

Server: `laya-serve` on `127.0.0.1:8765` from the local snapshot
(`$PWD/.failure-loc/serve_local.py` in the coding-decisions checkout).

## 5. Metrics

- **macro-F1** over the 8 classes (primary), plus weighted F1, accuracy, and the
  per-class P/R/F1 table.
- confusion matrix, and a secondary macro-F1 over classes with support ≥ 5
  (drops `contract`).
- bootstrap (10k resamples, **clustered by project**) → 95% CI for macro-F1 and
  for `Δ = Laya − best baseline`.
- every table split by setting (in-project LOO / cross-project).

## 6. Bar

> **Pass** = Laya macro-F1 ≥ best baseline + **0.10** **and** ≥ **0.55**
> absolute, in the in-project setting, with the bootstrap CI on Δ excluding 0.

If B2 already scores ≥ 0.70, the rule asks for ≥ 0.80 — if that looks
unreachable, the finding is "the embedding kNN is the right tool", which is a
valid outcome and gets recorded as one.

## 7. Deliverables and budget

| step | time | output |
|---|---|---|
| collect + scrub + leakage report | 0:20 | `.work/memory-types.jsonl`, counts |
| B1 + metrics harness | 0:30 | baselines table |
| B2 embedding kNN | 0:20 | strong baseline or a recorded fallback |
| Laya run (490 calls) | 0:15 | `.work/memory-type-laya.jsonl` |
| analysis + `docs/memory-type-results.md` | 0:35 | verdict + tables |

Total ≈ 2 h. Files: `scripts/eval_memory_type.py` (subcommands `collect`,
`baselines`, `laya`, `report`), `docs/memory-type-results.md`.

## 8. Outcomes

- **Pass** → a `totem type` *audit* in dsh-totem: memories whose stored type
  disagrees, surfaced for review. Advisory only. ~2 h more.
- **Fail** → record it, and if anyone wants type suggestions later, point them
  at B2 (the embedding kNN), not at Laya.

## 9. Risks

- **Label noise.** Types were chosen by agents over months; the eval measures
  agreement with those choices, not truth. Say so in the results.
- **Class imbalance.** Three classes have ≤ 5 examples, so macro-F1 on them is
  noise. That is why the CI is reported and a support ≥ 5 variant exists.
- **Tag leakage** would inflate every method equally — but it hides the real
  difference, so it is stripped.
- **Project vocabulary.** In-project LOO can be gamed by project-specific words;
  the cross-project number is the check on that.
