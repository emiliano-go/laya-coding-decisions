"""Evaluate a decision model on the held-out encoder-style decision set.

  uv run --no-sync python eval_dec.py <model> [dev.jsonl]

Reports per-question balanced accuracy, separation, and ECE for noul questions
and MAE for score questions.
"""
import json
import random
import sys
from collections import defaultdict

import laya

random.seed(5)
model = sys.argv[1] if len(sys.argv) > 1 else "/data/laya-ft/out-dec"
dev = sys.argv[2] if len(sys.argv) > 2 else "/data/laya-ft/dev_dec.jsonl"
agent = laya.load(model, device="cuda")
print("loaded", model, flush=True)

rows = [json.loads(line) for line in open(dev)]
by_q = defaultdict(list)
for row in rows:
    state = row["state"]
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    is_slice = row.get("slice") == "memory-ablation"
    for qid, g in gold.items():
        by_q[qid].append((state, questions[qid], g, is_slice))


def balanced(preds, golds):
    tp = sum(1 for p, g in zip(preds, golds) if p and g)
    tn = sum(1 for p, g in zip(preds, golds) if (not p) and (not g))
    fp = sum(1 for p, g in zip(preds, golds) if p and not g)
    fn = sum(1 for p, g in zip(preds, golds) if (not p) and g)
    tpr = tp / (tp + fn) if tp + fn else 0
    tnr = tn / (tn + fp) if tn + fp else 0
    return (tpr + tnr) / 2


def auc(scores, labels):
    pos = sum(1 for label in labels if label)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos_rank = sum(r for r, label in zip(ranks, labels) if label)
    return (pos_rank - pos * (pos + 1) / 2) / (pos * neg)


def ece(probs, correct, bins=10):
    if not probs:
        return None
    total = len(probs)
    error = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo < p <= hi) or (b == 0 and p <= 0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        error += (len(idx) / total) * abs(conf - acc)
    return error


def brier(probs, labels):
    return sum((p - label) ** 2 for p, label in zip(probs, labels)) / len(probs)


def report(label, qid, cases):
    preds, golds, confs, ps = [], [], [], []
    for state, qdef, g, _is_slice in cases:
        res = agent.system_one(state, {"q": qdef})
        ans = res["answers"]["q"]
        if qid == "bug_risk":
            p = ans.get("score", 0)
            gold_level = g["label"]
            preds.append(round(p) == gold_level)
            golds.append(True)
            ps.append(p)
            confs.append(ans.get("confidence", 0.5))
        else:
            p = ans.get("noul", 0.5)
            preds.append(p > 0.5)
            golds.append(g["label"] == 1)
            ps.append(p)
            confs.append(ans.get("confidence", max(p, 1 - p)))
    if qid == "bug_risk":
        acc = sum(preds) / len(preds)
        mae = sum(abs(p - g["label"]) for (_, _, g, _), p in zip(cases, ps)) / len(ps)
        q_ece = ece(confs, preds)
        print(f"{label:34s} n={len(cases):6d} exact={acc:.3f} mae={mae:.3f} ece={q_ece:.3f}")
        return
    bal = balanced(preds, golds)
    acc = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)
    sep_t = sum(p for p, g in zip(ps, golds) if g) / max(1, sum(golds))
    sep_f = sum(p for p, g in zip(ps, golds) if not g) / max(1, len(golds) - sum(golds))
    q_auc = auc(ps, golds)
    q_ece = ece(confs, preds)
    q_brier = brier(ps, golds)
    auc_txt = f"{q_auc:.3f}" if q_auc is not None else "n/a"
    print(
        f"{label:34s} n={len(cases):6d} acc={acc:.3f} bal={bal:.3f} "
        f"auc={auc_txt} ece={q_ece:.3f} brier={q_brier:.3f} "
        f"sep={sep_t:.2f}/{sep_f:.2f} gap={sep_t - sep_f:+.2f}"
    )


for qid, cases in sorted(by_q.items()):
    report(qid, qid, cases)
    slice_cases = [case for case in cases if case[3]]
    if slice_cases and len(slice_cases) != len(cases):
        report(f"{qid} [memory-ablation]", qid, slice_cases)
