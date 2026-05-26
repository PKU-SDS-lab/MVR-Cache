#!/usr/bin/env python3
"""
Smoke-test loading a Lightning checkpoint for ResumeFriendlyREINFORCE and running 1 forward pass.

Example:
  python scripts/checkpoint_smoketest.py \
    --ckpt /path/to/last.ckpt \
    --pairs_json /path/to/train_pairs.json \
    --max_len 512 \
    --max_segments 8 \
    --gpu 0
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from tensordict.tensordict import TensorDict

# Ensure HF mirror is set early (matches RL4COTrainer.py behavior)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from RL4COTrainer import ResumeFriendlyREINFORCE  # noqa: E402
from embedding_model import EmbeddingModel  # noqa: E402
from MaxSimGenerator import MaxSimGenerator  # noqa: E402
from MaxSimEnv import MaxSimEnv  # noqa: E402
from AdaptedPointerNetworkPolicy import AdaptedPointerNetworkPolicy  # noqa: E402


def load_pairs_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"pairs_json must be a non-empty list: {path}")
    return data


def load_prompts_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        raise ValueError(f"prompts file is empty: {path}")
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt file")
    ap.add_argument("--pairs_json", default=None, help="Pairs JSON with sentence_1/sentence_2/correct")
    ap.add_argument("--prompts_txt", default=None, help="Prompts txt (one prompt per line)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_segments", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    if (args.pairs_json is None) == (args.prompts_txt is None):
        raise SystemExit("Provide exactly one of --pairs_json or --prompts_txt")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] device={device}")

    # Build embedding model + generator/env/policy exactly like training
    embedding_model = EmbeddingModel(device=device)

    if args.pairs_json is not None:
        pairs = load_pairs_json(args.pairs_json)
        gen = MaxSimGenerator(
            prompts=None,
            pairs=pairs,
            max_len=args.max_len,
            embedding_model=embedding_model,
            seed=123,
        )
    else:
        prompts = load_prompts_txt(args.prompts_txt)
        gen = MaxSimGenerator(
            prompts=prompts,
            pairs=None,
            max_len=args.max_len,
            embedding_model=embedding_model,
            seed=123,
        )

    env = MaxSimEnv(generator=gen, max_segments=args.max_segments, embedding_model=embedding_model, device=device)
    policy = AdaptedPointerNetworkPolicy(env, embedding_dim=768, hidden_dim=768, max_segments=args.max_segments)

    # Initialize model skeleton, then load checkpoint weights
    model_kwargs = dict(
        env=env,
        policy=policy,
        baseline="rollout",
        train_data_size=2560,
        val_data_size=2560,
        batch_size=args.batch,
        dataloader_num_workers=0,
        optimizer_kwargs={"lr": 1e-4},
    )

    # Preferred: Lightning-style load (will call on_load_checkpoint hooks)
    print(f"[LOAD] Loading checkpoint: {args.ckpt}")
    # Fix for PyTorch 2.6+ weights_only=True default
    model = ResumeFriendlyREINFORCE.load_from_checkpoint(args.ckpt, **model_kwargs, strict=False, weights_only=False)
    model.strict_loading = False
    model.to(device)
    model.eval()

    # Generate a small batch and run the policy forward + env reward
    with torch.no_grad():
        td = gen._generate(args.batch)
        # policy expects td tensors on its target device; move explicitly
        td = td.to(device)
        out = model.policy(td, env=model.env, phase="test", select_best=False, decode_type="sampling")
        actions = out["actions"]
        reward = env._get_reward(td, actions)

    print("[OK] Forward pass succeeded.")
    print(f"[SHAPES] actions={tuple(actions.shape)} reward={tuple(reward.shape)}")
    print(f"[REWARD] mean={reward.mean().item():.6f} min={reward.min().item():.6f} max={reward.max().item():.6f}")


if __name__ == "__main__":
    main()


