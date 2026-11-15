"""Head-only checkpoint merge / task arithmetic for Laya.

Merges only the non-encoder tensors (`head.*`, `scorer.*`, `type_emb.*`,
`act_head.*`) so the shared encoder is untouched.

  python scripts/merge_heads.py --a out-m5mem4-head --b out-m6-head \
      --base out-multi --mode task --alpha 0.4 --out /data/laya-ft/merge-tmp
"""
from __future__ import annotations

import argparse
import os
import shutil

from safetensors.torch import load_file, save_file


def merge(a_dir: str, b_dir: str, base_dir: str, mode: str, alpha: float, out: str, only_scorer: bool = False) -> None:
    a = load_file(os.path.join(a_dir, "model.safetensors"))
    b = load_file(os.path.join(b_dir, "model.safetensors"))
    base = load_file(os.path.join(base_dir, "model.safetensors")) if mode == "task" else None
    merged = {}
    for key, value in a.items():
        touches = key.startswith("scorer.") if only_scorer else not key.startswith("encoder.")
        if not touches or key not in b:
            merged[key] = value
            continue
        if mode == "task":
            assert base is not None and key in base
            merged[key] = value + alpha * (b[key] - base[key])
        else:
            merged[key] = (1.0 - alpha) * value + alpha * b[key]
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    for name in ("encoder", "tokenizer"):
        os.symlink(os.path.join(a_dir, name), os.path.join(out, name))
    for name in ("rl_agent_config.json",):
        shutil.copy2(os.path.join(a_dir, name), os.path.join(out, name))
    save_file(merged, os.path.join(out, "model.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="/data/laya-ft/out-m5mem4-head")
    parser.add_argument("--b", default="/data/laya-ft/out-m6-head")
    parser.add_argument("--base", default="/data/laya-ft/out-multi")
    parser.add_argument("--mode", default="task", choices=["task", "interp"])
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--out", default="/data/laya-ft/merge-tmp")
    parser.add_argument("--only-scorer", action="store_true")
    args = parser.parse_args()
    merge(args.a, args.b, args.base, args.mode, args.alpha, args.out, args.only_scorer)
    print(f"wrote {args.out} ({args.mode} alpha={args.alpha} scorer_only={args.only_scorer})")


if __name__ == "__main__":
    main()
