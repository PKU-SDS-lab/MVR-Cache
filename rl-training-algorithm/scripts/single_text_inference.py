#!/usr/bin/env python3
"""
Single-text inference for the segmentation policy (φ(text)).

This loads a Lightning checkpoint for `ResumeFriendlyREINFORCE`, then runs
`model.policy.forward_single(...)` on ONE text input and prints:
- pointer positions
- pointer tokens
- reconstructed segments

Example:
  python scripts/single_text_inference.py \
    --ckpt /data2/ali/checkpoints_seperate/last.ckpt \
    --text "how to learn pytorch for deep learning. can you give me a tutorial on pytorch tensors?" \
    --gpu 2 \
    --max_len 512 \
    --max_segments 8 \
    --decode_type sampling
"""

import argparse
import os
import sys
from typing import List

import torch
from tensordict.tensordict import TensorDict

# Ensure the project root is on sys.path so `import RL4COTrainer` works regardless of cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure HF mirror is set early (matches RL4COTrainer.py behavior)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from embedding_model import EmbeddingModel  # noqa: E402
from AdaptedPointerNetworkPolicy import AdaptedPointerNetworkPolicy  # noqa: E402
from MaxSimEnv import get_segments_from_token_pointers  # noqa: E402


def _load_texts(args) -> List[str]:
    if args.text is not None:
        return [args.text]
    with open(args.text_file, "r", encoding="utf-8") as f:
        texts = [ln.strip() for ln in f if ln.strip()]
    if not texts:
        raise SystemExit(f"Empty text_file: {args.text_file}")
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt file")
    ap.add_argument("--text", default=None, help="Single text to segment")
    ap.add_argument("--text_file", default=None, help="Path to txt file (one text per line)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_segments", type=int, default=8)
    ap.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        choices=["greedy", "sampling"],
        help="Decoding strategy for pointer selection",
    )
    ap.add_argument("--debug", action="store_true", help="Enable debug_records from forward_single")
    args = ap.parse_args()

    if (args.text is None) == (args.text_file is None):
        raise SystemExit("Provide exactly one of --text or --text_file")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] device={device}")

    texts = _load_texts(args)

    # Build embedding model and a minimal env-like holder for policy init.
    embedding_model = EmbeddingModel(device=device)
    env_stub = type("EnvStub", (), {})()
    env_stub.reward_lm = embedding_model
    env_stub.device = device

    policy = AdaptedPointerNetworkPolicy(
        env_stub,
        embedding_dim=768,
        hidden_dim=768,
        max_segments=args.max_segments,
        policy_mode="separate",
    )

    # Load ONLY the policy weights from the Lightning checkpoint.
    # This avoids calling Lightning/rl4co `setup()` (which expects an env with dataset()).
    print(f"[LOAD] Loading policy weights from checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", {})
    pol_sd = {}
    for k, v in sd.items():
        if not k.startswith("policy."):
            continue
        # Drop env-attached tensors from training (not needed for inference)
        if k.startswith("policy.env."):
            continue
        pol_sd[k[len("policy.") :]] = v

    missing, unexpected = policy.load_state_dict(pol_sd, strict=False)
    print(f"[LOAD] policy missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if len(unexpected) > 0:
        print(f"[LOAD] unexpected_keys (first 10): {unexpected[:10]}")
    policy.to(device)
    policy.eval()

    # Embed single texts -> build td_single in the expected single-text format
    tok = embedding_model.tokenizer
    token_embeds = embedding_model.get_token_embeddings(
        texts,
        max_length=args.max_len,
        device=device,
        return_device=device,
    )
    emb = token_embeds["last_hidden_state"]  # [B, L, D]
    input_ids = token_embeds["input_ids"]  # [B, L]
    attn = token_embeds["attention_mask"]  # [B, L]
    length = attn.sum(dim=1)

    td_single = TensorDict(
        {
            "token_embeddings": emb,
            "attention_mask": attn.to(torch.bool),
            "input_ids": input_ids,
            "length": length,
        },
        batch_size=[len(texts)],
        device=device,
    )

    with torch.no_grad():
        out = policy.forward_single(
            td_single,
            decode_type=args.decode_type,
            key_suffix=None,
            side="S",
            debug=bool(args.debug),
            debug_topk=10,
            debug_n_samples=min(2, len(texts)),
        )

    actions = out["actions"]  # [B, K]
    print(f"[OK] actions shape={tuple(actions.shape)} log_likelihood shape={tuple(out['log_likelihood'].shape)}")

    # Print per-sample segments
    for i, text in enumerate(texts):
        ptrs = [int(x) for x in actions[i].tolist()]
        # Clamp pointers to the actual effective length range for printing
        la = int(length[i].item())
        ids = input_ids[i, :la]
        toks = tok.convert_ids_to_tokens(ids.tolist())
        ptrs_clamped = [min(max(0, p), len(toks) - 1) for p in ptrs]
        ptr_tok = ", ".join([f"{p}:{toks[p]}" for p in ptrs_clamped if 0 <= p < len(toks)])

        segs = get_segments_from_token_pointers(
            text,
            ids,
            ptrs_clamped,
            tok,
            skip_cls=True,
            pointers_are_end_boundaries=True,
        )

        print("\n" + "=" * 80)
        print(f"[SAMPLE {i}] text={text!r}")
        print(f"[SAMPLE {i}] pointers={ptrs_clamped}")
        print(f"[SAMPLE {i}] pointer_tokens={ptr_tok}")
        print(f"[SAMPLE {i}] segments ({len(segs)}):")
        for j, s in enumerate(segs):
            print(f"  {j+1:02d}: {s!r}")

    if args.debug and isinstance(out.get("info"), dict) and "debug_records" in out["info"]:
        print("\n[DEBUG] First few debug_records:")
        for r in out["info"]["debug_records"][:10]:
            print(r)


if __name__ == "__main__":
    main()


