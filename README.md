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

- **perturbed** — a correct patch with a rule-guided bug injected
  (`requirements_met=1`, `bug_risk=3`);
- **cross-instance** — another task's correct patch (`requirements_met=0`,
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
general decision ability and runs ~3x faster). A full fine-tune from the base
checkpoint forgets the reasoning heads — see `RESULTS.md`.

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
