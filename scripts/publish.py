#!/usr/bin/env python
"""Publish a coding-decisions checkpoint to the Hugging Face Hub.

Dry-run by default; pass --yes to upload. Writes a model card from the
checkpoint's `rl_agent_config.json` and any `RESULTS.md` next to it.

    python scripts/publish.py --repo emiliano-go/laya-coding-decisions --checkpoint /data/laya-ft/out-coding-decisions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_CARD = """\
---
library_name: laya
tags: [decision-engine, coding-agent, judge]
---

# {repo}

A Laya decision checkpoint that judges coding-agent turns: given a task and a
patch it scores **requirements_met** (does the change do what was asked) and
**bug_risk** (does it introduce a bug). Trained with the
[coding-decisions recipe](https://github.com/emiliano-go/laya) for the encoder /
hestia decision engine (`POST /v1/systemone`).

- Backbone: `{encoder}`
- max_len / head_max_len: {max_len} / {head_max_len}
- Fine-tuned: {fine_tuned} from `{model_name}`

## Use

```sh
pip install laya
laya-serve   # then POST /v1/systemone with {{state, questions}}
```

Point encoder's `decision.baseUrl`, or hestia's Decision settings, at the server.

## Intended use

Judging a completed agent turn. Scores are pooled by the client's decision rule
(veto + weighted/log-odds aggregation); do not treat a single score as a gate.
"""


def build_card(checkpoint: Path, repo: str) -> str:
    cfg: dict = {}
    config_path = checkpoint / "rl_agent_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
    return MODEL_CARD.format(
        repo=repo,
        encoder=cfg.get("encoder", "?"),
        max_len=cfg.get("max_len", "?"),
        head_max_len=cfg.get("head_max_len", "?"),
        fine_tuned=cfg.get("fine_tuned", False),
        model_name=cfg.get("model_name", "?"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. org/laya-coding-decisions")
    parser.add_argument("--checkpoint", required=True, help="local checkpoint directory")
    parser.add_argument("--yes", action="store_true", help="actually upload (otherwise dry-run)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "model.safetensors").exists():
        sys.exit(f"not a Laya checkpoint (no model.safetensors): {checkpoint}")

    card = build_card(checkpoint, args.repo)
    files = sorted(p.name for p in checkpoint.iterdir())
    print(f"repo:       {args.repo}")
    print(f"checkpoint: {checkpoint}")
    print(f"files:      {files}")

    if not args.yes:
        print("\n[dry-run] would create the repo, write README.md, and upload the checkpoint.")
        print("re-run with --yes to publish.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    (checkpoint / "README.md").write_text(card)
    api.upload_folder(repo_id=args.repo, folder_path=str(checkpoint), repo_type="model")
    print(f"published: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
