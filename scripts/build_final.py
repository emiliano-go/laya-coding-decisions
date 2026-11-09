"""Combine m5f + teacher bug/scope + teacher answer rows into one training file.

Answer rows are oversampled hard so the undertrained `answer_quality` head gets
signal against 255k patch rows.

Usage: python3 build_final.py [--out /data/laya-ft/train_m5final.jsonl]
"""
import argparse
import collections
import json
import os
import random

BASE = os.environ.get("LAYACD_DATA", "/data/laya-ft")


def load(path, split="train"):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if split is None or r.get("split") == split:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BASE, "train_m5final.jsonl"))
    ap.add_argument("--teacher-repeat", type=int, default=3)
    ap.add_argument("--answer-repeat", type=int, default=10)
    ap.add_argument("--memory-repeat", type=int, default=8)
    ap.add_argument("--benign-repeat", type=int, default=5)
    args = ap.parse_args()
    random.seed(23)

    m5f = load(os.path.join(BASE, "train_m5f.jsonl"), split=None)
    teacher = load(os.path.join(BASE, "gen_deepseek/teacher_all.jsonl"))
    answers = load(os.path.join(BASE, "gen_deepseek/answers/rows_teacher.jsonl"))
    memory = load(os.path.join(BASE, "gen_deepseek/memory_cf.jsonl"))
    benign = load(os.path.join(BASE, "gen_deepseek/benign/rows_teacher.jsonl"))

    combined = m5f + teacher * args.teacher_repeat + answers * args.answer_repeat + memory * args.memory_repeat + benign * args.benign_repeat
    random.shuffle(combined)
    with open(args.out, "w") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")
    kinds = collections.Counter(r.get("kind", "base") for r in combined)
    print(json.dumps({
        "m5f": len(m5f),
        "teacher": len(teacher),
        "answers": len(answers),
        "memory": len(memory),
        "benign": len(benign),
        "teacher_repeat": args.teacher_repeat,
        "answer_repeat": args.answer_repeat,
        "memory_repeat": args.memory_repeat,
        "benign_repeat": args.benign_repeat,
        "total": len(combined),
        "kinds": dict(kinds),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
