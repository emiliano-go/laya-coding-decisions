# Failure-driven localization — run record (2026-11-16)

**Verdict: negative. Gate 1 fails, so Phase B (Host/Tool/UI) was not built.**
The feature does not beat the deterministic evidence already present in a pytest
failure, and the shortlist that would have to carry it is the actual bottleneck.

Companion documents: [`failure-localization-plan.md`](failure-localization-plan.md)
(evaluation design) and [`failure-localization-impl.md`](failure-localization-impl.md)
(engineering breakdown).

## Corpus

Built with `scripts/build_failure_loc.py` from
`/data/laya-ft/swe-smith/hard.json` (26 instances):

| outcome | count |
|---|---:|
| usable | **21** |
| unsupported language (Go: caddy, nats-server, go-critic, goatcounter) | 4 |
| image absent (MONAI) | 1 |
| gold sites | **34** |
| instances with a source traceback frame (`frame` stratum) | 14 |
| assertion-only (`assert-only` stratum) | 7 |

Each instance is the SWE-smith testbed with the injected bug applied, the real
`pytest -q -rA` failure captured, and the failing test files moved to `oracle/`
before candidates are extracted. Gold is the candidate whose span contains the
patch's added lines; the bug patch is never shown to the model.

Container: `laya-serve` on `127.0.0.1:8765`, CPU, checkpoints served from the
local HF snapshot (see *Infrastructure*). One `POST /v1/systemone` per
instance/config; ~1.0 s p50 latency.

## Results (k=12, failure window 2000 chars, 21 instances / 34 sites)

| config | top1 | top3 | MRR | instance top1 |
|---|---:|---:|---:|---:|
| `L` task + failure | **0.235** | **0.471** | **0.390** | 0.381 |
| `H3` deterministic shortlist order (frame + BM25) | 0.176 | 0.412 | 0.340 | 0.286 |
| `H1` innermost source frame | 0.176 | 0.412 | 0.340 | 0.286 |
| `H2` BM25 only | 0.147 | 0.382 | 0.306 | 0.238 |
| `L0` task text only (the old issue probe) | 0.147 | 0.294 | 0.284 | 0.238 |
| `L1` failure only | 0.176 | 0.265 | 0.289 | 0.286 |

Paired instance bootstrap (10k resamples, site-level MRR):

| comparison | delta | 95% CI |
|---|---:|---|
| `L − H3` | +0.068 | [−0.023, 0.185] |
| `L − L0` | +0.127 | [−0.030, 0.279] |
| `L − L1` | +0.092 | [−0.024, 0.214] |

`L` is ahead of every baseline, but no interval excludes zero: 34 sites over 21
instances cannot resolve a 0.07–0.13 effect.

### By stratum

| config | frame top1 (n=14) | assert-only top1 (n=7) |
|---|---:|---:|
| `L` | 0.250 | 0.214 |
| `H3` | 0.150 | 0.214 |
| `L0` | 0.100 | 0.214 |

The niche gate is the important null: in the assertion-only stratum — the only
place Laya has room to beat the traceback — it ties the deterministic order
exactly (0.214 vs 0.214).

### Ranking when the shortlist already contains the answer

Conditional on the gold candidate surviving retrieval (16 instances), top1:
`L` 0.500, `L0` 0.385, `H3` 0.375, `H1` 0.375, `H2` 0.312. Paired discordance for
`L vs H3` is 3 wins / 1 loss / 17 ties (exact sign test p = 0.31). A real signal
may exist, but this corpus cannot distinguish it from noise.

## Retrieval — the binding constraint

| k | R@k (sites) | R@k (instances) |
|---:|---:|---:|
| 8 | 0.559 | 0.619 |
| 12 | 0.647 | 0.762 |
| 16 | 0.676 | 0.762 |
| 24 | 0.676 | 0.810 |
| 40 | 0.735 | 0.857 |

12 of 34 gold sites miss at k=12. **Eight of those twelve sit in files that
never appear in any traceback frame** (`sqlparse/filters/output.py`,
`dask/array/linalg.py`, `scrapy/exporters.py`, `conan/.../pkg_type.py`,
`nikola/plugins/task/authors.py`, `exceptiongroup/_exceptions.py`): the failure
output simply does not name them, and the issue text does not either. The other
four are in a frame file but lose the BM25 tie inside a large module.

Improving this is not a model problem. Making the shortlist better needs a
signal that a pytest failure does not carry (import graph, coverage, or the
agent's own exploration), which is a different feature.

Sensitivity: the top1 is not monotonic in `k` (`k=8` 0.265, `k=12` 0.235,
`k=16` 0.265); larger shortlists raise recall but add distractors. A 4000-char
failure window does not help (0.235 top1, 0.390 MRR).

## Gates

| gate | requirement | measured | pass |
|---|---|---|---|
| 1 retrieval | R@12 ≥ 0.85 | 0.647 | **no** |
| 2 ranking | Δtop1 vs H3 ≥ 0.10, CI > 0 | +0.059, CI [−0.023, 0.185] | **no** |
| 3 failure signal | Δ ≥ 0.10 | +0.127 MRR / +0.088 top1, CI crosses 0 | no* |
| 4 niche | assert-only L ≥ 0.40, H3 ≤ 0.20 | 0.214 vs 0.214 | **no** |

\* the point estimate clears the bar, the interval does not; recorded as a fail
because the gate requires the effect to be separated from zero.

Decision: **stop**. No route, tool, or UI changes; `POST /totem/laya/localize`
stays the manual diff-narrowing endpoint and Laya stays off by default. This is
the fourth independent result against Laya as an autonomous or semi-autonomous
signal (gate A/B, head fine-tune, memory warm start, failure localization).

## Deviations from the plan (and why)

1. **Gold resolved against candidate spans, not AST.** AST gold produces ids the
   extractor never emits for overload sets and nested functions. Span resolution
   guarantees `gold ∈ candidates`; AST is no longer used at all.
2. **BM25 normalised to [0,1]** before weighting. Raw BM25 grows with corpus
   size and was swamping the frame evidence on large repos.
3. **Failure text is focused, not tailed.** Taking the tail of a pytest run kept
   a plugin's teardown traceback and pushed the real assertion out of the
   window. `focus_failure` keeps the failing node ids, the first two failure
   blocks (head + frame marker), and the short summary, and the model sees that
   at query time. Raw output is stored for frame parsing.
4. **Third-party frames dropped** (`site-packages`, `/opt`, `/usr`) and
   `strip_root` rewrites only path starts — a blind `replace("/testbed/","")`
   turned `/testbed/opt/miniconda3/envs/testbed/lib` into a bogus repo path.
5. **`conan`/`hydra` were re-run with the cacheprovider enabled** because
   `-p no:cacheprovider` crashes `pytest_snail`; without the retry both produced
   plugin errors instead of test failures.

## Reproduction

```sh
# server (local snapshot, no network)
.laya-venv/bin/python .failure-loc/serve_local.py     # from the workspace root

# corpus (~5 min, docker only for export + pytest)
python scripts/build_failure_loc.py --tasks /data/laya-ft/swe-smith/hard.json \
  --root .failure-loc --no-pull --resume
python scripts/build_failure_loc.py --root .failure-loc --recapture --tail 12000

# probe
python scripts/localize_failure.py --root .failure-loc --configs L,L0,L1 \
  --k 12 --tail 2000 --out .failure-loc/results.json

# unit rules
python scripts/localize_candidates.py --self-check
python scripts/test_localize_candidates.py
```

The dataset was written to `$PWD/.failure-loc` rather than `/data/laya-ft/failure-loc`
because the agent sandbox mounts `/data` read-only; readers of `/data` are
unaffected.

## Infrastructure note

`laya-serve` cannot use the existing HF cache directly: `refs/main` points to a
root-only snapshot (`aa8c91ca`) while the full bundle lives in another revision
(`1c5edc17`), so it fails with `Subfolder 'multilingual' not found`, and
`HF_HUB_OFFLINE=1` still calls `repo_info`. The server is therefore started with
an explicit `Router(models={...})` over local paths (`serve_local.py`,
`max_loaded=3`).

## Call-shape experiments (2026-11-16, follow-up)

The first probe asked one 12-way choice per instance. Since Laya is cheap, the
follow-up tested whether the *shape* of the call was leaving signal on the table:
short option labels (the head budget is shared, `head_max_len=192`, with
`per = max(4,(192-16)//n)` tokens per option), code snippets as option text,
one call per candidate, one call per pair, and the other two checkpoints.

Same corpus, same shortlists (k=12 unless noted), english checkpoint unless noted:

| mode | top1 | top3 | MRR | cond. top1 | cond. MRR |
|---|---:|---:|---:|---:|---:|
| `H3` deterministic order | 0.176 | 0.412 | 0.340 | 0.231 | 0.421 |
| `choice_long` (original) | **0.235** | **0.471** | **0.385** | **0.308** | **0.480** |
| `choice_short` (h1..h12 labels) | 0.235 | 0.412 | 0.384 | 0.308 | 0.479 |
| `choice_short_code` (snippets as options) | 0.176 | 0.324 | 0.332 | 0.231 | 0.411 |
| `pointwise` noul, 12 calls | 0.088 | 0.176 | 0.225 | 0.115 | 0.270 |
| `pairwise` + Bradley-Terry, 28 calls, k=8 | 0.147 | 0.441 | 0.331 | 0.227 | 0.451 |
| `typed-decisions` checkpoint | 0.235 | 0.412 | 0.372 | 0.308 | 0.538 |
| `multilingual` checkpoint | 0.059 | 0.265 | 0.228 | 0.077 | 0.346 |

(`pairwise` is at k=8, where its `H3` control is 0.176 / 0.412 / 0.353 and only
14 instances / 22 sites survive retrieval.)

Findings:

1. **The option-budget hypothesis is dead.** Short labels change nothing
   (0.235 → 0.235). The model was not starved of per-option tokens.
2. **More calls do not help.** Pointwise per-candidate scoring is far worse
   (0.088): `noul` saturates around 0.5-0.85 and tracks code length more than
   relevance. Pairwise comparisons are worse than the deterministic order at the
   same `k` (0.147 vs 0.176 top1, 0.331 vs 0.353 MRR). The 12-way choice is the
   best shape available, not a compromise.
3. **Checkpoints are equivalent.** `typed-decisions` matches english on top1 and
   is slightly worse on MRR; `multilingual` collapses (0.059), as expected for a
   language checkpoint on code.
4. **Throughput.** On CPU the cost is ~1 s per request regardless of batching:
   12 concurrent requests became one forward pass that still took 12.4 s
   (1 s/row). The published 33 ms figure is the GPU path. Batching is not a
   lever on this machine.

Net: the call shape and the checkpoint are exhausted; the residual +0.06 top1
over the deterministic order stays inside the noise band from the main run. The
only untried lever is retrieval, which is not a Laya change (coverage or import
graph from the failing test, plus a larger corpus for power).

## Hierarchical narrowing (2026-11-16, follow-up)

Was it implemented? **No.** The only shipped surface is the single-shot
`POST /totem/laya/localize` (rank the hunks of one diff) plus `laya_localize`.
The dir → file → section → function cascade existed only as a proposal.

`scripts/localize_narrow.py` now implements and evaluates three shapes over the
same corpus, all with short option labels and a beam of 3:

1. **pure cascade** — directory → file → class → function → 10-line window,
   using "part k of N" sections when a level has more than 48 children;
2. **hybrid** — cascade only to the top-3 files, then rank functions inside each
   chosen file with the usual BM25 + 8-way choice;
3. **window probe** — with the gold function given, rank 10-line windows.

| shape | calls | top1 | top3 | MRR |
|---|---:|---:|---:|---:|
| pure cascade | 165 | 0.000 | 0.059 | 0.128 |
| hybrid (files, then in-file) | 129 | 0.029 | 0.088 | 0.112 |
| `H3` deterministic flat shortlist | 0 | **0.176** | **0.412** | **0.340** |
| single 12-way choice (earlier) | 21 | 0.235 | 0.471 | 0.385 |

File-level accuracy for the hybrid cascade: **top1 0.190, top3 0.286**; when the
top-1 file was right, the in-file ranking put the gold function first only 1 time
in 4.

Failure mode is visible in the picks. The cascade is drawn to *scripts and build
tooling* that the failure text mentions — `bench_omd.py`, `qps-bench-server.py`,
`hatch_build.py`, `dodo.py`, `_version.py`, `generate_version.py` — over library
modules, and when it does land in a 300-400 function file it has to choose among
arbitrary contiguous "part k of 12" sections, which is close to a coin flip. The
two levels that behaved well were narrow ones: `monkeytype/typing.py` at 0.987
and `exceptiongroup/_exceptions.py` at 0.787.

The window probe is underpowered and inconclusive: only 17 gold functions had at
least two 10-line windows, and the top1 of 0.176 is at chance for that fan-out.

Conclusion: the hierarchy is worse than both the flat model call and plain
deterministic retrieval. A cascade only helps if the coarse stage is a better
*retriever* than BM25 (coverage, import graph, call graph); Laya is not. This is
the same conclusion as the call-shape follow-up, reached from the other
direction.

## N-best patch selection (2026-11-16, follow-up)

The strongest remaining reading of "Laya catches bugs" is `requirements_met`
(AUC 0.82 on dev). The one shape where that would pay off is choosing among
candidate patches — and the data for it was already on disk: SWE-Rebench ships
6,291 tasks with execution-verified candidate diffs (4,808 with 4-12 candidates,
base rate ~0.46 correct).

`scripts/rerank_rebench.py`, english checkpoint, one call per task:

| setup | tasks | pick@1 | random | shortest | longest | correct mass |
|---|---:|---:|---:|---:|---:|---:|
| ~8.6 candidates, stat synopsis | 600 | 0.465 | 0.482 | 0.492 | 0.483 | 0.480 |
| 4 candidates, + first added lines, requirements framing | 201 | 0.657 | 0.665 | 0.682 | 0.657 | 0.664 |
| 4 candidates, pointwise `noul` requirements question | 60 | 0.633 | 0.654 | 0.683 | 0.650 | — |

The probability mass on correct candidates equals the base rate in every setup:
the distribution over candidates is uniform. Laya does not merely fail to beat
the simplest heuristics; it contributes nothing.

This also corrects the record on the old 12/14 funnel catch. `RESULTS.md` notes
that catch came from `requirements_met` plus the deterministic verify lane, not
from `bug_risk`. The verification is binary ("does this patch meet the request"),
and a binary judgement does not order candidates — which is exactly what the
uniform distributions above show.

Conclusion: patch selection is dead for this model, not for lack of candidates or
budget. If a selector is wanted, it has to be a model that reads code.
