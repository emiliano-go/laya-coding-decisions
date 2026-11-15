"""Weight-Space Programming probe for Laya's decision heads.

Reads the scorer's hidden representation for a question, computes a paired
direction (e.g. level3 - level0), applies a rank-1 edit to the final scorer
layer, and sweeps the edit strength. This is a WSP-style edit: the LLM chooses
the direction and the evaluator accepts only measured improvements.

The edit touches only `scorer.3.weight` (1 x 1024), not the encoder.

  python scripts/wsp_probe.py --question bug_risk --limit 1200
  python scripts/wsp_probe.py --question bug_class --target-class wrong-comparison
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import laya
from laya.common import QTYPES, build_sequence, collate_items, temp_bucket


def load_rows(path: str, question: str, limit: int):
    rows = []
    for line in open(path):
        row = json.loads(line)
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        if question not in gold:
            continue
        questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        if question not in questions:
            continue
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        rows.append((state, questions[question], int(gold[question]["label"])))
        if limit and len(rows) >= limit:
            break
    return rows


def capture(agent, rows, question: str):
    tok, cfg = agent.tok, agent.cfg
    max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["h"] = inputs[0].detach()

    handle = agent.model.scorer[3].register_forward_hook(hook)
    hidden, labels = [], []
    batch_size = 8
    qtype = QTYPES["score" if question == "bug_risk" else "choice"]
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            items = []
            for state, qdef, _label in batch:
                q = agent._to_internal(qdef)
                seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
                items.append({"ids": seq, "markers": markers, "qtype": qtype})
            b = collate_items([items], tok.pad_token_id)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=agent.dtype):
                agent.model(
                    b["input_ids"].to("cuda"), b["attention_mask"].to("cuda"),
                    b["marker_pos"].to("cuda"), b["marker_mask"].to("cuda"), b["qtype"].to("cuda"),
                )
            h = captured["h"].float().cpu().numpy()
            for index, (_state, _qdef, label) in enumerate(batch):
                k = int(b["marker_mask"][index].sum())
                hidden.append(h[index, :k].copy())
                labels.append(label)
    finally:
        handle.remove()
    return hidden, np.asarray(labels)


def probs_from_hidden(hidden, weight, bias, temperature):
    logits = np.asarray([h @ weight + float(bias[0]) for h in hidden])
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def auc(scores, positive):
    pos = scores[positive]
    neg = scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def risk_metrics(hidden, labels, weight, bias, temperature):
    p = probs_from_hidden(hidden, weight, bias, temperature)
    expected = (np.arange(p.shape[1])[None, :] * p).sum(1)
    exact = float((np.clip(np.round(expected), 0, 3) == labels).mean())
    mae = float(np.abs(expected - labels).mean())
    p3 = p[:, 3]
    threshold = float(((p3 >= 0.5).astype(int) * 3 == labels).mean())
    return {"exact": round(exact, 4), "mae": round(mae, 4), "p3_acc": round(threshold, 4), "auc": round(auc(p3, labels == 3), 4)}


def class_metrics(hidden, labels, weight, bias, temperature, keys):
    p = probs_from_hidden(hidden, weight, bias, temperature)
    pred = p.argmax(1)
    accuracy = float((pred == labels).mean())
    macro = float(np.mean([((pred == i) & (labels == i)).sum() / max(1, (labels == i).sum()) for i in range(len(keys))]))
    return {"acc": round(accuracy, 4), "macro": round(macro, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/laya-ft/out-m5mem4-head")
    parser.add_argument("--data", default="/data/laya-ft/dev_m6.jsonl")
    parser.add_argument("--question", default="bug_risk", choices=["bug_risk", "bug_class"])
    parser.add_argument("--target-class", default="wrong-comparison")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument("--direction", default="paired", choices=["paired", "state3", "state_mean"])
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    agent = laya.load(args.model, device="cuda")
    rows = load_rows(args.data, args.question, args.limit)
    rng = np.random.RandomState(0)
    rng.shuffle(rows)
    if args.question == "bug_risk":
        rows = [row for row in rows if row[2] in (0, 3)]
        if args.balance:
            zero = [row for row in rows if row[2] == 0]
            three = [row for row in rows if row[2] == 3]
            n = min(len(zero), len(three))
            rows = zero[:n] + three[:n]
            rng.shuffle(rows)
    print(f"rows {len(rows)} labels {dict(Counter(row[2] for row in rows))}")
    hidden, labels = capture(agent, rows, args.question)
    split = int(len(hidden) * args.split)
    train_h, train_y = hidden[:split], labels[:split]
    val_h, val_y = hidden[split:], labels[split:]

    weight = agent.model.scorer[3].weight.detach().float().cpu().numpy().reshape(-1).copy()
    bias = agent.model.scorer[3].bias.detach().float().cpu().numpy().copy()
    qtype = QTYPES["score" if args.question == "bug_risk" else "choice"]
    k = len(hidden[0])
    temperature = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
    print(f"temperature {temperature:.3f} k {k}")

    # Direction in the final scorer input space. For bug_risk use level3-level0;
    # for choice use target class minus "none" (or the first class).
    if args.question == "bug_risk":
        directions = {}
        if args.direction in ("paired",):
            directions["level3-level0"] = np.mean([h[3] - h[0] for h in train_h], axis=0)
        if args.direction in ("state3",):
            directions["state3"] = np.mean([h[3] for h, y in zip(train_h, train_y) if y == 3], axis=0) - np.mean([h[3] for h, y in zip(train_h, train_y) if y == 0], axis=0)
        if args.direction in ("state_mean",):
            directions["state_mean"] = np.mean([h.mean(0) for h, y in zip(train_h, train_y) if y == 3], axis=0) - np.mean([h.mean(0) for h, y in zip(train_h, train_y) if y == 0], axis=0)
    else:
        keys = list(agent._to_internal(rows[0][1])["crit"].keys())
        target = keys.index(args.target_class) if args.target_class in keys else len(keys) - 1
        base = keys.index("none") if "none" in keys else 0
        directions = {f"{args.target_class}-{keys[base]}": np.mean([h[target] - h[base] for h in train_h], axis=0)}

    print("baseline", risk_metrics(val_h, val_y, weight, bias, temperature) if args.question == "bug_risk" else class_metrics(val_h, val_y, weight, bias, temperature, list(agent._to_internal(rows[0][1])["crit"])))
    best = None
    for name, direction in directions.items():
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        scale = float(np.mean([np.linalg.norm(h) for h in train_h]))
        unit = direction / norm / max(scale, 1e-9)
        print(f"direction {name} norm {norm:.3f} scale {scale:.3f}")
        for alpha in (-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64):
            edited = weight + alpha * unit
            if args.question == "bug_risk":
                metrics = risk_metrics(val_h, val_y, edited, bias, temperature)
                print(f"  alpha {alpha:>3} " + " ".join(f"{key}={value}" for key, value in metrics.items()))
                score = metrics["exact"]
            else:
                keys = list(agent._to_internal(rows[0][1])["crit"].keys())
                metrics = class_metrics(val_h, val_y, edited, bias, temperature, keys)
                print(f"  alpha {alpha:>3} " + " ".join(f"{key}={value}" for key, value in metrics.items()))
                score = metrics["acc"]
            if best is None or score > best[0]:
                best = (score, alpha, name, edited)

    if best is None:
        raise SystemExit("no direction")
    score, alpha, name, edited = best
    print(f"best alpha {alpha} {name} score {score:.4f}")
    if args.out:
        shutil.copytree(args.model, args.out, dirs_exist_ok=True)
        weights = load_file(f"{args.out}/model.safetensors")
        key = "scorer.3.weight"
        weights[key] = torch.tensor(edited, dtype=weights[key].dtype).reshape(1, -1)
        save_file(weights, f"{args.out}/model.safetensors")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
