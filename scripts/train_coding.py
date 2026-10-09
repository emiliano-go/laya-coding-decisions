"""Single-GPU fine-tune of laya on coding decisions.

Adapted from the laya typed-decisions notebook (RLCD policy gradient with
proper-scoring rewards + soft cross-entropy), minus DDP so it runs on one GPU.

Efficiency:
  - length-bucketed batching (near-zero padding; random order pads 1.3-1.8x)
  - tokenized items cached to disk
  - bf16 or fp16 autocast
  - optional gradient checkpointing
  - a checkpoint is written after every epoch

Run through the uv project:
  uv run --no-sync python train_coding.py --data train_hf.jsonl --out out-hf-A \
    --epochs 3 --micro-batch 4 --grad-accum 4 --trainable all --optim 8bit --dtype bf16
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def build_training_item(tok, cfg, state, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    else:
        return None
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    k = len(render_options({"t": t, "crit": crit}))
    seq, markers = build_sequence(
        tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, cfg["max_len"], cfg["head_max_len"]
    )
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": label}


def load_items(tok, cfg, path, cache, recache):
    if os.path.exists(cache) and not recache:
        print(f"loading cached items from {cache}")
        return torch.load(cache, weights_only=False)
    items = []
    for line in open(path):
        row = json.loads(line)
        state = row["state"]
        if isinstance(state, str) and state.strip().startswith(("{", "[")):
            try:
                state = json.loads(state)
            except Exception:
                pass
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            if qid in gold:
                it = build_training_item(tok, cfg, state, q, gold[qid])
                if it:
                    items.append(it)
    torch.save(items, cache)
    print(f"cached {len(items)} items to {cache}")
    return items


def make_batches(items, batch_size, window, seed):
    """Length-bucketed batches: sort by length, shuffle within a window to keep
    sources mixed, then group. Near-zero padding because a batch shares a length."""
    rng = random.Random(seed)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for start in range(0, len(order), window):
        segment = order[start : start + window]
        rng.shuffle(segment)
        order[start : start + window] = segment
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def save_model(model, tok, cfg, out):
    os.makedirs(out, exist_ok=True)
    was_training = model.training
    model.eval()
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out, "encoder"))
    tok.save_pretrained(os.path.join(out, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-coding-decisions"
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if was_training:
        model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_hf.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1.0e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-max-len", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0, help="cap number of items (0 = all)")
    ap.add_argument("--model-id", default="convaiinnovations/laya")
    ap.add_argument(
        "--trainable",
        default="all",
        help="all | head | topN (freeze everything except the last N encoder layers and the head)",
    )
    ap.add_argument("--optim", default="8bit", help="adamw | 8bit")
    ap.add_argument("--dtype", default="bf16", help="bf16 | fp16")
    ap.add_argument("--bucket-window", type=int, default=1000)
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--recache", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model_dir = args.model_id
    if not os.path.isdir(model_dir):
        print("resolving model...")
        model_dir = snapshot_download(args.model_id)
    else:
        print(f"loading from local checkpoint {model_dir}")
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = args.head_max_len

    cache = f"{args.data}.items.{args.max_len}.{args.head_max_len}.pt"
    all_items = load_items(tok, cfg, args.data, cache, args.recache)
    if args.limit:
        all_items = all_items[: args.limit]
    print(f"  {len(all_items)} training sequences")
    by_type = {}
    for it in all_items:
        by_type[it["qtype"]] = by_type.get(it["qtype"], 0) + 1
    print(f"  by qtype (choice=0,score=1,noul=2): {by_type}")

    print("building model...")
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    if args.trainable == "head":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif args.trainable.startswith("top"):
        n = int(args.trainable[3:])
        layers = list(getattr(model.encoder, "layers", []))
        for layer in layers[: max(0, len(layers) - n)]:
            for p in layer.parameters():
                p.requires_grad = False
        embeddings = getattr(model.encoder, "embeddings", None)
        if embeddings is not None:
            for p in embeddings.parameters():
                p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({args.trainable})")

    if not args.no_gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
        print("gradient checkpointing: on")
    else:
        print("gradient checkpointing: off")
    model.to(device)
    model.train()

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": enc_params, "lr": args.lr_encoder}, {"params": head_params, "lr": args.lr_head}]
    if args.optim == "8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    use_bf16 = args.dtype == "bf16"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    print(f"dtype: {args.dtype}")

    updates_per_epoch = max(1, len(all_items) // (args.micro_batch * args.grad_accum))
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    print(f"updates/epoch~{updates_per_epoch} total_updates={total_updates}")

    sigma_start, sigma_end = 0.4, 0.1
    t0 = time.time()
    for epoch in range(args.epochs):
        batches = make_batches(all_items, args.micro_batch, args.bucket_window, seed=42 + epoch)
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        running = 0.0
        n_batches = 0
        progress = epoch / max(1, args.epochs - 1)
        sigma = sigma_start + (sigma_end - sigma_start) * progress
        last_reset = time.time()
        for bi, indices in enumerate(batches):
            chunk = [all_items[j] for j in indices]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=autocast_dtype):
                logits, act = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / args.grad_accum

            scaler.scale(loss).backward()
            accum += 1
            if accum % args.grad_accum == 0 or bi == len(batches) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * args.grad_accum
            n_batches += 1
            if n_batches % 500 == 0:
                elapsed = time.time() - last_reset
                last_reset = time.time()
                print(
                    f"  epoch {epoch + 1}/{args.epochs} batch {n_batches}/{len(batches)} "
                    f"loss {running / n_batches:.4f} reward {r.mean().item():.3f} "
                    f"mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
                    f"{n_batches / (time.time() - t0):.1f} batch/s",
                    flush=True,
                )
        elapsed = time.time() - t0
        print(
            f"=== epoch {epoch + 1} done in {elapsed:.1f}s, {len(all_items) / elapsed:.1f} seq/s, "
            f"avg loss {running / max(1, n_batches):.4f}, peak mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB ===",
            flush=True,
        )
        save_model(model, tok, cfg, args.out)

    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
