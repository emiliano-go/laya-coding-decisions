"""Per-option logit-bias sweep for bug_class and bug_risk.

This is the classifier-style WSP fix: the model's probabilities are p; adding
bias b to the option logits is equivalent to p'_i ~ p_i * exp(b_i). We sweep a
prior-correction strength tau and report macro/accuracy (bug_class) or exact/MAE
(bug_risk). Nothing is trained.

  python scripts/bias_sweep.py --question bug_class --limit 1907
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

import laya


def rows_with(path: str, question: str, limit: int):
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


def metrics(probs, labels, question, keys):
    if question == "bug_class":
        pred = probs.argmax(1)
        acc = float((pred == labels).mean())
        recalls = [float(((pred == i) & (labels == i)).sum() / max(1, (labels == i).sum())) for i in range(len(keys))]
        return {"acc": round(acc, 4), "macro": round(float(np.mean(recalls)), 4)}, recalls
    expected = (np.arange(probs.shape[1])[None, :] * probs).sum(1)
    exact = float((np.clip(np.round(expected), 0, 3) == labels).mean())
    mae = float(np.abs(expected - labels).mean())
    p3 = float(((probs[:, 3] >= 0.5).astype(int) * 3 == labels).mean())
    return {"exact": round(exact, 4), "mae": round(mae, 4), "p3": round(p3, 4)}, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/laya-ft/out-m5mem4-head")
    parser.add_argument("--dev", default="/data/laya-ft/dev_v7.jsonl")
    parser.add_argument("--train", default="/data/laya-ft/train_v7.jsonl")
    parser.add_argument("--question", default="bug_class", choices=["bug_class", "bug_risk"])
    parser.add_argument("--limit", type=int, default=1907)
    args = parser.parse_args()

    agent = laya.load(args.model, device="cuda")
    rows = rows_with(args.dev, args.question, args.limit)
    print(f"{args.question}: dev rows {len(rows)} labels {dict(Counter(r[2] for r in rows))}")

    # Priors from the training split.
    priors = Counter()
    for line in open(args.train):
        row = json.loads(line)
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        if args.question in gold:
            priors[int(gold[args.question]["label"])] += 1
    total = sum(priors.values())
    k = len(priors)
    prior = np.array([priors.get(i, 0) / total for i in range(k)])
    print("priors", dict(priors), "normalized", np.round(prior, 3).tolist())

    probs, labels = [], []
    for state, qdef, label in rows:
        result = agent.system_one(state, {"q": qdef})
        answer = result["answers"]["q"]
        if args.question == "bug_class":
            keys = list(agent._to_internal(qdef)["crit"].keys())
            p = np.array([answer["probabilities"].get(key, 0.0) for key in keys], dtype=float)
        else:
            p = np.array([answer["probabilities"].get(str(i), 0.0) for i in range(4)], dtype=float)
        probs.append(p)
        labels.append(label)
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    baseline, recalls = metrics(probs, labels, args.question, range(k))
    print("baseline", baseline, "recalls", None if recalls is None else [round(v, 3) for v in recalls])

    logprior = np.log(np.clip(prior, 1e-9, None))
    best = None
    for sign in (-1, 1):
        for tau in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
            bias = sign * tau * logprior
            adjusted = probs * np.exp(bias)[None, :]
            adjusted = adjusted / adjusted.sum(1, keepdims=True)
            value, _ = metrics(adjusted, labels, args.question, range(k))
            score = value["macro"] if args.question == "bug_class" else value["exact"]
            print(f"  sign {sign:+d} tau {tau:>4} {value}")
            if best is None or score > best[0]:
                best = (score, sign, tau, bias, value)
    print("best", best[0], "sign", best[1], "tau", best[2], best[4])
    out = {
        "question": args.question,
        "bias": best[3].tolist(),
        "sign": best[1],
        "tau": best[2],
        "metrics": best[4],
        "baseline": baseline,
        "prior": prior.tolist(),
    }
    path = f"/data/laya-ft/bias_{args.question}.json"
    with open(path, "w") as sink:
        json.dump(out, sink, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
