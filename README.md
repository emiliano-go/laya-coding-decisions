# coding-decisions recipe

Train **Laya** into a judge for coding-agent turns: given a task and a patch,
score `requirements_met` and `bug_risk`. Built for the encoder / hestia decision
engine, which posts a strict state to `POST /v1/systemone`.

> **Status:** research recipe. The published weights are not ready yet; the
> current best judge is still the base `convaiinnovations/laya` ("out-multi").
> See `RESULTS.md`.

## Layout

```
scripts/
  fetch_dec.py                       build data/cache3/*.jsonl from public sources
  gen_dataset_v2.py                  build train/dev/test_<suffix>.jsonl
  train_coding.py                    single-GPU fine-tune (RLCD + soft CE)
  eval_dec.py                        judge-dev metrics: balanced acc, AUC, ECE, MAE
  smoke_test.py                      stdlib-only invariant check for the builder
  publish.py                         upload a checkpoint to the Hugging Face Hub
```

## Install

```sh
uv sync          # or: pip install -e .
```

## Data

First build the source cache (needs network + `datasets`):

```sh
LAYACD_DATA=/data/laya-ft python scripts/fetch_dec.py
```

`gen_dataset_v2.py` consumes `$LAYACD_DATA/cache3/{swe_smith_traj,swe_rebench_traj,gold,bigcodebench}.jsonl`
(each row: `task`, `diff`, `correct`, `extra`) and writes
`$LAYACD_DATA/{train,dev,test}_m5d.jsonl` (`LAYACD_DATA` defaults to `/data/laya-ft`).
Each row is `{id, state, questions, gold}` where `state = {task, memory, answer,
changes, toolCalls}` and `gold` carries the `requirements_met` / `bug_risk` labels.

Hard-negatives that make the judge useful:

- **perturbed**: a correct patch with a rule-guided bug injected
  (`requirements_met=1`, `bug_risk=3`);
- **cross-instance**: another task's correct patch (`requirements_met=0`,
  `bug_risk=0`).

`memory` is derived from the **task only** (a requirement restatement plus a
task-independent gotcha, chosen by task hash), so memory never predicts the
label; ~20% of rows carry `memory: []`. `smoke_test.py` asserts these invariants.

## Train

```sh
uv run --no-sync python scripts/train_coding.py \
  --data "$LAYACD_DATA/train_m5d.jsonl" --out "$LAYACD_DATA/out-coding-decisions" \
  --epochs 1 --micro-batch 2 --grad-accum 8 --trainable all \
  --optim 8bit --dtype bf16 --model-id convaiinnovations/laya
```

`--trainable head` freezes the encoder (recommended while iterating: it preserves
general decision ability and runs about 3x faster). A full fine-tune from the base
checkpoint forgets the reasoning heads; see `RESULTS.md`.

## Evaluate

```sh
uv run --no-sync python scripts/eval_dec.py "$LAYACD_DATA/out-coding-decisions" "$LAYACD_DATA/dev_m5d.jsonl"
```

Gates: `requirements_met` balanced accuracy ≥ 0.80 and AUC ≥ 0.75; `bug_risk`
exact ≥ 0.60 and MAE ≤ 0.70; ECE ≤ 0.15.

## Serve

Local checkpoint (stdlib shim shipped with encoder):

```sh
python /path/to/encoder/script/laya-server.py --port 8765 --model "$LAYACD_DATA/out-coding-decisions"
```

Then point the client at it: `decision.baseUrl = http://127.0.0.1:8765`
(encoder `decision` config, or hestia's Decision settings panel).

The in-repo `laya serve` path needs the model registered and a local-path option;
see `REGISTRATION.md` for the exact change.

## Publish

```sh
python scripts/publish.py --repo emiliano-go/laya-coding-decisions --checkpoint "$LAYACD_DATA/out-coding-decisions"
```

Dry-run by default; `--yes` uploads.

## Background and plan

### Why this exists

The starting problem is slop: a coding agent produces output that looks plausible but is wrong, and nothing automatically catches it.

Two capabilities fight it. A decision engine scores every finished agent turn with a small local classifier (a Laya model); if the turn does not clear the bar the engine injects review feedback and the agent retries. A deterministic verify lane always catches hard failures independent of the model: hardcoded secrets, SQL or shell injection, destructive commands, and failing tests or typechecks. Totem memory stores durable decisions, gotchas, invariants, and constraints so an agent keeps hard-won knowledge across sessions.

The weak link is the classifier. A general Laya checkpoint ("out-multi") scores a coding turn without coding-specific training, so this repository exists to specialize Laya into a coding judge. It is the second of two shipping repos:

- `emiliano-go/hestia`: the product (web cockpit, reachable over a VPN, Python FastAPI plus React) with the encoder features migrated in; the encoder TUI was dropped.
- `emiliano-go/laya-coding-decisions`: this training kit.

They connect at runtime over HTTP: the product posts a turn state to the judge at `POST /v1/systemone`, and the judge returns typed answers.

### The plan

Product migration (done): decision engine, editing core with write grades, snapshots and a `/tmp` sandbox, session compaction and reminders, AGENTS.md hierarchy, provider key pools and small-model, memory auto-registration and export/import, and a registry migration to Turso.

Training (this repo): build the dataset, fine-tune, evaluate, serve, publish. The model program tracks M1 (veto, log-odds pooling, default-deny, verify lane), M2 (real diffs, execution evidence, grounding), M3 (harness labels and diagnostics), M5 (retrain), and the v4 label fix that decouples `requirements_met` from `bug_risk`.

Gates: judge-dev (requirements balanced accuracy at least 0.80 and AUC at least 0.75; `bug_risk` exact at least 0.60 and MAE at most 0.70; ECE at most 0.15) and the offline funnel bench (catch at least 13 of 14, false pass at most 1 of 14).

### Expected results

| checkpoint | funnel catch | false pass | requirements bal | bug_risk exact |
|---|---|---|---|---|
| out-multi (base) | 12/14 | 2/14 | 0.783 | 0.011 |
| out-m5c (v4 full fine-tune) | 10/14 | 4/14 | 0.760 | 0.469 |
| out-m5c-head (head-only, in flight) | target at least 12/14 | target at most 2/14 | target at least 0.78 | target at least 0.60 |

Success: the head-only run preserves the base checkpoint's reasoning heads while adapting the bug-risk head; if it beats the base on the funnel without losing requirements accuracy it becomes the candidate judge, then the memory-aware run follows, weights are published, and the model is registered in Laya.

Fallback: keep the base checkpoint as the judge and lean on the deterministic verify lane, calibration, and a real harness-label flywheel (labels from actual task outcomes instead of the synthetic dataset). The failed full fine-tunes already showed the synthetic dataset alone is not enough.

What we learned (also in `RESULTS.md`): full fine-tuning from the base checkpoint degrades the funnel every time; decoupling the labels did not recover it; improving `bug_risk` exact did not improve the funnel. The funnel is driven mostly by `requirements_met` plus the verify lane.

### Relation to Totem

Totem is the memory layer and it is central, not an add-on.

- Storage: the product uses the real `totem_mcp` Python package (schema v9 with verification records), one database per project clone at `<clone>/.totem/totem.db` plus a user database. An earlier encoder integration used a TypeScript Totem port that lagged at schema v3; the merge removes that drift.
- Judge input: before scoring, the decision engine reads ranked project memory (matched by task and changed path) into the judge state, so memory gives the judge authority the diff alone cannot, such as an invariant or constraint the change violates.
- Judge output: after a verdict the engine writes the decision back onto the memories the turn produced (`metadata.decision`), so memory is rated by outcomes.
- File changes: writing, editing, or patching a file auto-registers an implementation memory through Totem's locator dedup (`register_file_write`).
- Distillation: the memory writer runs on the small model.

The symbiosis is a loop: the agent writes memory; the decision engine reads memory to judge the turn; the verdict is written back to memory; future turns get better context from better-rated memories.

Deferred: Totem's verification layer (`verified_at`, `verified_commit`, a context boost for fresh verifications) could give the judge a first-class trust signal, but the current training carries plain memory only (`id`, `type`, `title`, `statement`).
