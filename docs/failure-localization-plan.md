# Failure-driven bug localization — plan

Status: **run and rejected** (2026-11-16). Owner: Emiliano. Gate 1 failed:
R@12 = 0.647 vs the required 0.85, so no productization followed. Results and
the as-run deviations: [`failure-localization.md`](failure-localization.md).
Scope: offline evaluation only, no agent runs, no training, no gate changes.

## 0. What we are actually testing

Today's number (top1 0.30, top2 0.68, MRR 0.578 on 100 mutation rows) ranks
**2–4 hunks of one diff**. The proposed task ranks **functions across a whole
repo** after a test failure. Those are different units at different difficulty,
so "beat 0.30" is not a valid bar — a repo-scale task could legitimately score
lower than the hunk task while still being the more useful tool.

The real question is narrower and falsifiable:

> Once a test fails, does Laya rank the buggy function better than the
> deterministic evidence already in the failure output?

A pytest traceback names the file and line of the innermost source frame. When
that frame is inside the buggy function, a five-line heuristic is already
near-perfect and Laya can only add noise. Laya's only defensible niche is the
**assertion-only** failure: no source frame, so the ranker must reason from the
test name, the assertion diff, and the task text. That is the hypothesis, and
the evaluation is built to isolate it (see §4 strata).

`scripts/localize_probe.py` is the closest precedent: repo-wide functions,
issue-text-only shortlist of 10, **retrieval recall 0.333**, top5 0.5. The
bottleneck was retrieval, not ranking. Adding the failure output is aimed at
exactly that bottleneck, which is why §1.3 gates on recall before anything else.

## 1. Dataset

### 1.1 Corpus

`/data/laya-ft/swe-smith/hard.json` — 26 instances, of which **21 are Python**
(5 Go: caddy, nats-server, go-critic, goatcounter, and atlas materialized
earlier). 25/26 docker images are already local, so no network is needed.

Six Python repos are already materialized under `/data/laya-ft/swe-smith/`.
The rest are re-exported from their images, which is why this step is "docker
cp + pytest", not "run the agent".

Estimated size: 21 instances × ~2.4 bug sites ≈ **45–55 gold sites**, which is
the right sample size for a site-level metric (instance-level n=21 is too small
to resolve a 0.10 delta).

Stretch only if the result lands within noise of a gate: the `SWE-smith`
dataset on HF has ~10k instances with the same `patch` field; the builder below
takes any instance list, so widening is a data cost, not new code.

### 1.2 Per-instance artifacts

`/data/laya-ft/failure-loc/<instance_id>/`

| file | content | visible to Laya |
|---|---|---|
| `task.json` | SWE-smith row (patch, FAIL_TO_PASS, PASS_TO_PASS) | no |
| `repo/` | testbed with the bug patch applied, `.git` committed | yes (candidates only) |
| `failure.txt` | raw `pytest -q -rA` output for FAIL_TO_PASS at the buggy commit | yes |
| `gold.json` | gold sites derived from `patch` | no |
| `candidates.json` | frozen candidate list (see §2) | yes |

Build with `scripts/build_failure_loc.py`, reusing `run_swe_smith.py`'s
`export_testbed`, `apply_bug_patch`, `sanitize`, `unsupported`, and
`run_pytest`. Two changes: keep `.git` (needed for `git grep`-style path
resolution later) is **not** required, and store the failure output instead of
discarding it.

```sh
python scripts/build_failure_loc.py \
  --tasks /data/laya-ft/swe-smith/hard.json \
  --root  /data/laya-ft/failure-loc \
  --limit 21 --tail 8000
```

Validity conditions per instance (skip, with a reason, otherwise):

1. bug patch applies (`git apply`);
2. baseline FAIL_TO_PASS actually fails — the bug is live;
3. at least 2 function candidates survive §2 (a 1-candidate "ranking" is noise).

### 1.3 Gold

Never show `patch` to Laya. Derive gold offline:

1. parse `patch`, collect added-line numbers per destination file (the existing
   `patch_added_lines` in `localize_probe.py` already does this correctly,
   including `b/` prefixes and hunk offsets);
2. resolve each added line against the **frozen candidate spans** and pick the
   smallest containing span. (As run: AST gold was dropped — for overload sets
   and nested functions it produced ids the extractor never emits, so
   `gold ∉ candidates`. Spans guarantee the invariant.)
3. `primary` = the function with the most added lines (tie → earliest);
   `sites` = the full set, for multi-site scoring;
4. module-level or class-body mutations (added lines outside any function) →
   gold is the **module** candidate, which §2 always emits as `path::<module>`.

`gold.json` = `{instance_id, primary: "<rel>::<qualname>", sites: [...],
files: [...], stratum: "frame"|"assert-only"}`.

## 2. Candidate extraction and retrieval

Two stages, because a repo has 28–2,957 functions (iniconfig 28, tenacity 384,
typeguard 694, oauthlib 918, pdfminer 882, Red-DiscordBot 2,957; pandas and
dask are larger). A single choice question cannot carry that: choice options
share one `head_max_len` budget, so hundreds of labels leave a handful of
tokens each.

### Stage 1 — deterministic shortlist (no model)

One function:

```
score(fn) = 6·frame_hit + 3·file_in_traceback + 2·path_token_match
          + 1.5·bm25(failure_text, fn.text) + 1·bm25(task_text, fn.text)
```

- `frame_hit`: the exact `file:line` of a non-test traceback frame falls inside
  the function span (strongest signal, deliberately dominant).
- `file_in_traceback`: the function's file appears in any frame (including test
  files' imports — a mutated module often shows up in the import chain).
- `bm25`: standard BM25 over identifier/docstring tokens; reuse
  `localize_probe.shortlist`, which is already a BM25-ish scorer.

Keep `k = 12` by default, evaluate `k ∈ {8, 12, 16}`. Exclude `tests/`,
`test_*.py`, `conftest.py`, `setup.py`, `docs/`, `vendor/`, `.git/`,
`node_modules/`. Cap the walk at 3,000 `.py` files and 5 s.

### Stage 2 — candidate text

Frozen format, identical in the probe and in the Host (parity test in §6):

```
<relpath>:<qualname>(<args>) — <first docstring line or first body line, ≤60>
```

For nesting, `qualname` is dotted (`SpooledIOBase.tell`). Include the module
candidate `path::<module>` only when gold allowed it, i.e. always emit it, with
text `path (module level)`.

### Extraction parity

The probe can use `ast`. The Host (TS, in the sandbox) cannot, so the shipped
extractor is regex-based:

- candidate start: `^\s*(?:async\s+)?def\s+(\w+)\s*\(`, plus enclosing `class`
  stack from `^\s*class\s+(\w+)`;
- span: next `def`/`class` at indent ≤ the candidate's indent, else EOF;
- args: text between the parentheses on the `def` line (multi-line signatures
  degrade to the first line — acceptable).

The probe must generate candidates with **the same regex rules**, and a test
asserts ≥99% identical candidate ids on the six materialized repos (AST is used
only for gold). Otherwise the offline number does not describe the shipped
feature.

## 3. Query construction

One `POST /v1/systemone`, one `choice` question:

```json
{
  "state": {
    "task": "<problem_statement, truncated to 400 chars>",
    "changes": [{"tool": "pytest", "output": "<failure.txt tail>"}]
  },
  "questions": {
    "loc": {
      "type": "choice",
      "instructions": "A test is failing. Which function most likely contains the bug?",
      "criteria": {"<candidate id>": "<candidate text>"}
    }
  }
}
```

Notes:

- the failure tail goes in `changes[].output` because it is the output of a
  command; the canonical state order (`task, memory, changes, answer,
  toolCalls, grounding`) stays untouched;
- tail budget: keep the **last** 2,000 chars (server-side 2k cap) but evaluate
  `--tail {1000, 2000, 4000}` — pytest puts the assertion and traceback near
  the end, and the 2k cap may cut the frame list on multi-test failures;
- criteria ≤ 160 chars each (§2), 12 of them;
- `MAX_STATE_CHARS` is 50k and `MAX_QUESTIONS` 64, so there is headroom.

Ablations that must be run, same candidates, same k:

| id | state |
|---|---|
| `L` | task + failure |
| `L0` | task only (isolates the failure signal; ≈ today's `localize_probe`) |
| `L1` | failure only (isolates the task text) |

`L0` vs `L` is the headline result: it is the only comparison that answers
"does the failure output help at all".

## 4. Evaluation protocol

### Baselines (all deterministic, same shortlist)

| id | rule |
|---|---|
| `RND` | analytic: 1/k |
| `H1` | innermost non-test frame's enclosing candidate |
| `H2` | BM25(failure + task) top-1 |
| `H3` | frame if any, else H2 (the strong baseline to beat) |

### Metrics

- **site-level** (n ≈ 45–55): top1, top3, MRR over gold `sites`, clustered by
  instance;
- **instance-level** (n = 21): top1/top3 = any gold site in the top-k;
- **retrieval recall@k** of the shortlist (`R@8/12/16`) — the gate in §5;
- latency p50/p95 of the HTTP call;
- every number split by `stratum ∈ {frame, assert-only}`.

### Statistics

21 instances, so no point estimates alone. Paired bootstrap over **instances**
10,000 resamples → 95% CI for `Δ = L − H3` on site-level MRR and top1. Also
report the per-instance win/loss table, because a delta carried by two
outliers is not a feature.

### Outputs

`scripts/localize_failure.py --out /data/laya-ft/failure-loc/results.json`
plus a markdown table in `RESULTS.md` and `docs/failure-localization.md`.

## 5. Decision rule (all four must hold)

1. **Retrieval**: `R@12 ≥ 0.85`, else stop — ranking cannot fix a shortlist
   that misses the bug.
2. **Ranking**: `L` beats `H3` on site-level top1 by **≥0.10 absolute**, and
   wins ≥14/21 instances, and the bootstrap 95% CI lower bound on `ΔMRR` > 0.
3. **Failure signal**: `L − L0 ≥ 0.10` top1 (otherwise the failure output is
   decoration and we are back to the rejected issue-text probe).
4. **Niche**: in `assert-only`, `L` top1 ≥ 0.40 while `H3` ≤ 0.20.

Pass all four → ship §6. Fail any → **stop Laya research** per the existing
decision, keep `POST /totem/laya/localize` as the manual hunk-narrowing tool,
and record the negative result in `RESULTS.md`. No fine-tune, no new gate arm,
no calibration work either way.

## 6. Productization (only after a pass)

Keep it manual, Laya off by default, never gating:

- `POST /totem/laya/localize` accepts `{task, failure, candidates:[{id,text}]}`;
  when `candidates` is present it ranks them directly and returns
  `{top, candidates:[{id,text,score}], stratum, ms}`. The existing `{task,diff}`
  path is unchanged (backward compatible, no version bump).
- `laya_localize` gains the same fields; the agent supplies candidates it
  already read, which avoids a filesystem walker in the Host.
- Laya tab: a **"Narrow from failure"** card next to "Narrow a diff" — paste
  the failing output, the Host scans the session cwd with the frozen §2 regex
  extractor (bounded: 3k files / 5 s / skip tests+venv), ranks top-12, renders
  `file:function · score` rows. Manual button only; no hook on mutating tools.
- tests: candidate-extractor golden file (iniconfig + tenacity), route
  validation (missing task / empty candidates → 400), and a `localize`
  regression that mocks the server.

## 7. Schedule (~4.5 h)

| # | step | time |
|---|---|---|
| 0 | start server (`LAYA_PORT=8765 laya-serve`), health check | 0:10 |
| 1 | `build_failure_loc.py`; materialize 21 repos + baselines + gold | 0:50 |
| 2 | candidate extractor + parity test vs AST gold | 0:40 |
| 3 | `localize_failure.py`: shortlist, baselines, Laya, strata | 1:10 |
| 4 | runs: `k × tail × {L,L0,L1}` (≈ 20 configs × 21 calls, ~33 ms each) | 0:40 |
| 5 | analysis, bootstrap, `RESULTS.md` + doc | 0:40 |
| 6 | productization *iff* pass | 1:00–1:30 |

Materialization dominates wall-clock; it is unattended.

## 8. Risks

- **Traceback leakage** (highest): if most failures carry a source frame, the
  task is solved by `H1` and the strata will say so. That is a legitimate
  finding — it kills the feature for the common case and leaves only the
  assertion-only niche, which §5.4 measures.
- **Small corpus**: mitigated by site-level n≈50 and instance-clustered
  bootstrap; do not quote a bare top1 without the CI.
- **Huge repos** (pandas, dask): stage-1 cap and test-file exclusion matter;
  log shortlist size and recall per instance so a bad repo is visible.
- **Multi-site gold**: one query, several right answers. Site-level top1 is
  naturally generous; the `sites`-based MRR is the conservative number.
- **Regex extractor drift**: the parity test is the only thing keeping the
  offline metric honest; it is not optional.

## 9. Explicitly out of scope

No new gate arms, no per-tool gating, no head training on these labels, no
calibration work, no forced memory. The gate A/B (17/21 both arms), the loc
head-only fine-tune (0.28/0.68 vs 0.30/0.68), and the memory warm-start arm all
stay closed.
