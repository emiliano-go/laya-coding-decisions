# Failure-driven localization — implementation plan

Status: Phase A **built and run** (2026-11-16); Phase B **not started** because
the gates failed. Companion to
[`failure-localization-plan.md`](failure-localization-plan.md) (the evaluation
design). This file is the engineering breakdown: files, interfaces, order, tests.

Two phases. **Phase A is unconditional and is the reference implementation**:
a stdlib-only Python extractor that defines the candidate and shortlist rules,
the dataset builder, and the probe. **Phase B is gated on Phase A passing** and
ports the frozen rules to the Host (TypeScript) plus the route, tool, and UI.

The ordering rule that makes the result honest: *the eval must run on the same
regex extraction the Host will ship*, not on an AST extractor that is more
accurate than what users get. AST is used only to compute gold.

---

## 1. Phase A — reference implementation and evaluation

### A1. `scripts/localize_candidates.py` (new, ~240 LOC, stdlib only)

Pure, importable, no network, no GPU. This module is the contract.

```python
Candidate  = {id, path, qualname, kind: "function"|"module", start, end, args, text}
Frame      = {file, line, func}          # relative paths, repo-root stripped
```

| function | behaviour |
|---|---|
| `strip_root(failure, repo)` | rewrite `/testbed`, absolute repo prefixes, and `.../repo/pkg/x.py` to `pkg/x.py` |
| `parse_frames(failure)` | both pytest traceback dialects: short (`pkg/x.py:42: in f`) and long (`File "pkg/x.py", line 42, in f`); preserves order; dedupes consecutive repeats |
| `parse_failed_tests(failure)` | `FAILED tests/test_x.py::test_y` and `assert ...` lines, kept as BM25 query text |
| `walk_repo(root, *, max_files=3000, max_seconds=5)` | sorted deterministic walk; skips `.git .venv venv env build dist node_modules docs vendor .tox .mypy_cache .ruff_cache`; skips `tests/`, `test_*.py`, `*_test.py`, `conftest.py`, `setup.py`; `.py` only |
| `extract_candidates(root, paths)` | the **frozen regex rules**: `^\s*(?:async\s+)?def\s+(\w+)\s*\(`, `^\s*class\s+(\w+)`; qualname from the class stack; span = next `def`/`class` at indent ≤ start, else EOF; text = `path:qualname(args) — <first docstring line>` capped 160; always appends one `path::<module>` candidate per file |
| `bm25(query, docs)` | k1=1.5, b=0.75, over `[A-Za-z_][A-Za-z0-9_]{2,}` tokens, lowercased |
| `rank_candidates(candidates, failure, task, k)` | the plan's formula: `6·frame_hit + 3·file_in_traceback + 2·path_token_match + 1.5·bm25_failure + 1·bm25_task`; stable sort, tie → shorter path; returns top-k + `stratum` |
| `build_choice(candidates, task, failure, tail=2000)` | the exact `/v1/systemone` body (§3), shared by the probe and asserted against the Host by a golden test |
| `shortlist_recall(candidates, gold, k)` | `R@k` for the gate |

CLI: `--dump <repo> [--out json]` (parity fixture), `--self-check` (runs the
unit assertions in-process so CI has no pytest dependency).

**Frozen-rule note:** the regexes, the exclusion list, the text format, and the
score weights are copied verbatim into TS in B1. Any change here invalidates
the parity fixture and must be recorded in `RESULTS.md`.

### A2. `scripts/build_failure_loc.py` (new, ~200 LOC)

Reuses `export_testbed`, `apply_bug_patch`, `sanitize`, `unsupported`,
`run_pytest` from `run_swe_smith.py` (import, do not copy). Per instance:

1. skip Go/JS/Cargo repos (`unsupported`); skip if the image is absent;
2. export testbed, apply bug patch, `git init`+commit, hide `FAIL_TO_PASS` files
   into `oracle/` **before** candidate extraction (so no test body is ever a
   candidate, and the walk matches the harness's pre-fix state);
3. run `pytest -q -rA` on FAIL_TO_PASS; require failure, else mark invalid;
4. write `failure.txt` (raw, tail-trimmed to `--tail 8000`), `gold.json`,
   `candidates.json`, `manifest.jsonl`;
5. `--resume` skips an instance whose `status: ok` row already exists. Never
   re-export an instance that is already materialized.

Gold (§1.3 of the evaluation plan) is computed with `ast` on the buggy tree,
never with the regex extractor, and is never written into `candidates.json`.

```sh
python scripts/build_failure_loc.py --tasks /data/laya-ft/swe-smith/hard.json \
  --root /data/laya-ft/failure-loc --limit 21 --tail 8000 --resume
```

### A3. `scripts/localize_failure.py` (new, ~260 LOC)

One run = one list of configs × instances, all cached to
`results.jsonl` so re-analysis never re-calls the server.

```sh
python scripts/localize_failure.py --root /data/laya-ft/failure-loc \
  --configs L,L0,L1 --k 12 --tail 2000 --baselines --strata --out results.json
```

- `L` = task + failure, `L0` = task only, `L1` = failure only (§3 of the eval plan);
- baselines `H1/H2/H3` computed with the same candidate list, no server call;
- metrics: site-level top1/top3/MRR, instance-level any-hit, `R@k`, latency;
- `--bootstrap 10000` → paired instance bootstrap CI for `L − H3` and `L − L0`;
- refuses to run if `/health` is down; prints the server's model id so the run
  is attributable.

Server: `LAYA_PORT=8765 laya-serve` (the endpoint the Host expects; currently
not running).

### A4. Analysis

`RESULTS.md` gains a `### Failure-driven localization` section with the two
tables (overall, by stratum) and the four gates from §5 of the eval plan,
explicitly marked pass/fail. `docs/failure-localization.md` holds the full run
record: corpus hash, config grid, per-instance table, CI.

---

## 2. Phase B — ship it (only after all four gates pass)

### B1. `src/localize-candidates.ts` (new, ~260 LOC)

Pure module, no `ctx`, no I/O except `walkProject`. Exports:

```ts
export interface Candidate { id: string; path: string; qualname: string; text: string; line: number; end: number; kind: 'function' | 'module' }
export interface LocalizeFrame { file: string; line: number }

export function parseFrames(failure: string, root: string): LocalizeFrame[]
export function extractCandidates(path: string, text: string): Candidate[]
export function rankCandidates(candidates: readonly Candidate[], failure: string, task: string, k: number): { ranked: Candidate[]; stratum: 'frame' | 'assert-only' | 'none' }
export function buildCandidateRequest(input: { task: string; failure: string; candidates: Candidate[]; tail?: number }): { state: unknown; questions: unknown }
export async function walkProject(root: string, options?: { maxFiles?: number; maxMs?: number }): Promise<{ path: string; text: string }[]>
```

`walkProject` is bounded: sorted `readdir` with `withFileTypes`, skip list
identical to A1, 3,000 files / 5 s hard stop, `readFile` in batches of 32 to
avoid unbounded memory, and it never follows symlinks outside the root
(`realpath` prefix check). On overrun it returns what it has plus a
`truncated` flag surfaced in the response.

This module is a **line-for-line port** of A1's rules. The unit tests are the
fixtures generated by `A1 --dump` for `iniconfig` and `tenacity`, checked in
under `test/fixtures/localize-candidates/` (~30 KB total, hand-trimmed to the
ids/texts that exercise every rule: module-level, class nesting, decorator,
async, multi-line signature, one-line body, nested function).

### B2. `src/laya.ts` (+~90 LOC, `localize()` untouched)

Add a sibling method rather than overloading the existing one, so the diff path
cannot regress:

```ts
export interface CandidateLocalizationResult {
  mode: 'candidates'
  top: string | null
  candidates: { id: string; text: string; path: string; score: number }[]
  stratum: 'frame' | 'assert-only' | 'none'
  considered: number
  truncated: boolean
  ms: number
}

async localizeCandidates(input: { task: string; failure: string; candidates: Candidate[]; truncated?: boolean }, signal?: AbortSignal): Promise<CandidateLocalizationResult | undefined>
```

`LocalizationResult` gains `mode: 'diff'` (additive; existing consumers read
`.hunks` and are unaffected). Reuse `healthy()`, the same timeout/abort
composition, and `rankLocalization`'s ranking shape where possible.

### B3. `src/routes.ts` (+~35 LOC, one dispatch)

`POST /totem/laya/localize` keeps its path and gains two optional fields:

```jsonc
{
  "task": "...",                 // required, ≤400 chars (canonical cap)
  "failure": "...",              // ≤8000, tail kept
  "project": "/abs/repo",        // GUI mode: Host extracts candidates
  "candidates": [{"id": "x", "text": "x", "path": "x"}], // agent mode
  "diff": "...", "changes": []   // existing diff mode
}
```

Dispatch: `candidates` present → candidate mode (no walk). Else `failure` +
absolute `project` → candidate mode with `walkProject` + `rankCandidates`.
Else → existing diff mode. Validation errors are `400` with the same error
shape as today; `candidates` longer than 16 is truncated (order preserved) and
flagged, never silently re-ranked.

`project` is validated exactly like the existing `/totem/memories` route
(`isAbsolute`, exists, is a directory) and is never written anywhere.

### B4. `src/laya.ts` tool (+~20 LOC)

`laya_localize` gains `failure`, `project`, and `candidates` parameters with the
same semantics; the tool picks candidate mode when either is supplied. The
agent path (it already has the repo and the output) sends `candidates`, so the
tool never walks the filesystem on its own.

### B5. Client (+~140 LOC)

- `assistant-controller.ts`: `LayaLocalizeResult` becomes
  `DiffLocalizeResult | CandidateLocalizeResult`; `localize(input)` takes the
  discriminated request; new `narrowFromFailure(project, task, failure)`.
- `AssistantSection.tsx`: a second card, **"Narrow from failure"**, under
  "Narrow a diff": a failure textarea, a project path field prefilled from
  `state.project` (editable), a button, and a result list of
  `path:qualname · 42.1%`. Errors render inline like the existing card.
- No new settings key. Laya stays off by default and nothing here is called
  automatically.

### B6. Tests

| file | tests |
|---|---|
| `test/localize-candidates.test.ts` (new) | extractor fixtures (both repos), frame parser dialects, BM25 determinism, query golden, walkProject caps/symlinks |
| `test/laya.test.ts` | `localizeCandidates` ranks a mocked response; diff path unchanged |
| `test/client.test.ts` | candidate result renders; controller sends `project` for GUI mode and `candidates` for agent mode |
| `scripts/test_localize_candidates.py` | same fixtures in Python; asserts Python↔TS parity files match |

### B7. Build and verify

```sh
bun test                       # from dsh-totem
bun run build                  # rebuilds lib/ (client bundle included)
```

Then in the running GUI: enable Laya in the Totem settings tab, paste a known
failing pytest output with the project path, confirm the ranking returns the
buggy function, and confirm "Narrow a diff" still works. No server restart is
needed for a plugin rebuild; the page must be refreshed.

---

## 3. Order, checkpoints, budget

| # | step | time | checkpoint |
|---|---|---|---|
| A1 | extractor + self-check | 1:00 | `--dump iniconfig` produces plausible ids |
| A2 | dataset build | 0:50 | 21 instances, `R@12` printed per instance |
| A3 | probe | 1:10 | `L0` roughly reproduces `localize_probe` (recall ~0.33) |
| A4 | runs + gate analysis | 0:40 | four gates evaluated, pass/fail recorded |
| B1 | TS port + fixtures | 0:50 | parity test green |
| B2–B4 | engine, route, tool | 0:45 | route 400 matrix green |
| B5 | UI card | 0:35 | manual ranking in the GUI |
| B6–B7 | tests, build, verify | 0:50 | `bun test`, refreshed GUI |

Phase A ≈ 3:40, Phase B ≈ 3:00, total ≈ 6:40. The eval plan's 3–5 h was
research only; if the budget is binding, cut in this order: `L1` ablation →
`k∈{8,16}` grid → the GUI card (ship the tool + route only).

Hard stop: if gate 1 (`R@12 ≥ 0.85`) fails, do **not** start Phase B. Record the
negative and stop, per the standing decision.

## 4. Implementation-specific failure modes

- **Regex/indent drift** (highest): a decorator line, a multi-line signature, or
  a `class` with a docstring can shift spans. Fixtures must include one of each;
  spans are advisory (used for `frame_hit`), so a ±3-line error degrades
  gracefully rather than corrupting the candidate list.
- **Traceback path forms**: `--tb=short` vs long, `.../repo/pkg/x.py`, `/testbed/pkg/x.py`,
  and Windows-style separators in pasted output. `strip_root` handles all four;
  the fixture covers them.
- **Host event-loop stalls**: `walkProject` on pandas/dask. Batched reads plus a
  hard time stop; the response reports `truncated`.
- **Criteria budget**: 16 candidates × 160 chars is already ~2.5k chars; the
  server's choice head shares `head_max_len`, so 16 is the cap and the probe
  tests 8/12/16 explicitly.
- **Duplicate ids**: `qualname` collisions (two functions of the same name in
  different scopes) are real. Ids are `path:qualname` and deduped by suffixing
  `#<line>`; the route rejects duplicates rather than silently ranking one.
- **Existing path regression**: `localize()` is not edited; the diff route and
  `laya_localize` without new fields take the exact current code path, guarded
  by a golden response test.

## 5. Definition of done

- Phase A: `build_failure_loc.py` + `localize_failure.py` run end to end on 21
  instances from a clean `/data` state; `results.json` reproducible from cached
  rows; `RESULTS.md` has the gate table with pass/fail; the four gates are
  applied, not narrated.
- Phase B (only on pass): `bun test` and `bun run build` green; the Python and
  TypeScript extractors produce identical candidate ids on the fixtures; the
  GUI card ranks a real failure and the diff card still works; the gate remains
  untouched and off by default.

## 6. Rollback

Phase B is additive: the route dispatch falls through to the current diff path
when the new fields are absent, `laya_localize` is unchanged without them, and
the UI card is a separate component. Reverting is a single commit; no state,
config, or database migration is involved.
