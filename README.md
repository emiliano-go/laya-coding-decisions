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

## Why we train Laya this way

### What we are doing with Laya

Laya is a small local decision model: a ModernBERT encoder (about 421M parameters) with typed decision heads that answer questions in three shapes (noul, score, choice). It is non-generative: it does not write prose, it returns a distribution over answer options plus a confidence. That matters here because the judge runs on every finished agent turn: a generative model judge would be slow, cost tokens per call, and vary run to run; Laya is local, deterministic enough to gate on, and answers in tens of milliseconds.

We are specializing it from a general decision model into a coding judge. Given a task and the patch an agent produced, it answers two questions:

- `requirements_met` (noul): does the change do what the task asked, including edge cases.
- `bug_risk` (score, 0 to 3): does the change introduce a bug or miss an edge case.

The product (encoder, and now hestia) posts a strict turn state to the judge at `POST /v1/systemone`; the judge returns typed answers; the client vetoes or flags the turn and feeds corrections back. The judge is the part that cannot be hand-written, so we train it.

### Why memory is part of the training, not just the prompt

The judge sees only a bounded window of the turn (512 tokens): the task, a short answer, and the diffs. That is often not enough to decide correctness. The requirement may be an invariant, a constraint, or a repo gotcha that is not visible in the diff. Totem memory holds exactly those durable facts, so the runtime feeds ranked project memory into the judge state as an authority signal.

The catch is that the base checkpoint was never trained with that memory field, so at runtime memory is out of distribution and effectively inert. Training with memory is what makes it usable: it teaches the model to read the memory and weigh it against the patch.

The training also has to prevent a shortcut. If memory correlated with the label, the model would learn the correlation instead of reading the patch (this is exactly what happened earlier with a synthesized "execution evidence" field, and the funnel collapsed). So memory in the dataset is derived from the task only, never from whether the patch is correct:

- it is a requirement restatement plus a task independent gotcha, chosen by task hash;
- the same task gets the same memory on its correct, perturbed, and wrong task rows;
- about one in five rows carries empty memory, so "memory present" never predicts a label.

The verdict is also written back onto the memories the turn produced, so memory is rated by outcomes. The loop is: the agent writes memory; the judge reads it; the verdict updates it; later turns get better context.

Totem specifics: the product uses the real `totem_mcp` package (schema v9, with verification records), one database per clone at `<clone>/.totem/totem.db`. An earlier encoder integration used a TypeScript Totem port that lagged at schema v3; the merge removes that drift. File writes also auto-register implementation memories through Totem's locator dedup, and the memory writer distills prose on the small model. Totem's verification layer (verified date, verified commit) is deferred as a future trust signal.

### Why this dataset

We need many labeled examples of "task plus patch is correct" and "task plus patch is buggy", with the two axes expressed independently. Real agent trajectories do not give that cleanly, so the dataset is synthesized from public sources:

- resolved labeled agent trajectories (SWE-smith, SWE-rebench), which give real tasks, real patches, and a resolution signal for `requirements_met`;
- gold patch instances (SWE-Gym, SWE-smith), known correct patches;
- BigCodeBench, function level canonical versus deterministic mutants, giving correct versus buggy pairs at small granularity.

We then add the hard negatives the judge must learn to separate:

- perturbed: a correct patch with a rule guided bug injected, so `requirements_met=1` but `bug_risk=3`; this is the "looks right but is buggy" class the funnel exists to catch;
- cross-instance: a different task's correct patch, so `requirements_met=0` and `bug_risk=0`; this is the "unrelated changes" class.

The label structure is deliberately decoupled. Earlier versions tied `bug_risk` to `requirements_met` (buggy only when the wrong task), which made the two heads redundant and the model degenerate to near constant outputs. The fixed dataset keeps four combinations, so `bug_risk` carries information that is independent of `requirements_met`.

Two more choices: the split is grouped by task hash (train, dev, test) to avoid leakage between splits, and a slice of dev is held out specifically for memory ablation, rows where the gold hinges on reading the memory.

The honest limitation: synthetic labels are not the real outcome distribution, so the fallback, if training stalls, is a harness-label flywheel that draws labels from actual task outcomes instead.

### Why this training method

The method is Laya's own typed-decisions recipe, adapted to one GPU.

- Start from an existing checkpoint, not from scratch. The base already knows the decision protocol and has reasoning heads; we are adapting it, not rebuilding it.
- Objective: contrastive reinforcement (RLCD) with proper scoring rewards plus a soft cross-entropy term. The judge emits a distribution over levels, and proper scoring rules (spherical and ranked probability score) reward a calibrated distribution rather than just the argmax; the soft cross-entropy keeps the head anchored to the target distribution. The RL part samples noisy logits, scores them with the proper reward, normalizes the advantage (GRPO style), and takes a policy-gradient step; the cross-entropy is added alongside it.
- Efficiency: length bucketed batching, a tokenized item cache, bf16 autocast, gradient checkpointing, 8-bit AdamW, cosine schedule, gradient clipping. It fits a 7.65 GB card; the full run is about 9 hours, a head-only run about 3.
- Fine-tuning strategy: full versus head-only. Full fine-tuning from the base checkpoint degrades the funnel: every full run lost catch and forgot the reasoning heads. Head-only freezes the encoder so the base reasoning is preserved while the heads adapt to the new labels. That is the current experiment.
- Calibration after training: temperature scaling per question type, plus runtime per-question Platt or isotonic calibration and a conformal threshold, because the funnel cares about calibrated confidence, not just the predicted level.

### What good looks like

Two gates, because the product metric is slop catch versus false pass, not raw accuracy. A model can have a better `bug_risk` exact and still a worse funnel (we measured that), so the funnel is the gate that matters.

- judge-dev: `requirements_met` balanced accuracy at least 0.80 and AUC at least 0.75; `bug_risk` exact at least 0.60 and MAE at most 0.70; ECE at most 0.15.
- funnel bench: catch at least 13 of 14 known bad cases and at most 1 false pass.

Measured so far: the base checkpoint catches 12 of 14 with 2 false passes; the v4 full fine-tune catches 10 of 14 with 4 false passes (worse), even though its `bug_risk` exact rose from 0.011 to 0.469. That gap is the reason the project is now on a head-only run. Details in `RESULTS.md`.
