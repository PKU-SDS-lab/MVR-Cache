"""
Quick debug tool: print MaxSimSplitter token-pointer segmentation for a single pair.

Example (manual texts):
  python benchmarks/debug_splitter_segments.py \
    --splitter-checkpoint /path/to/ckpt_or_dir \
    --device cuda \
    --text-a "..." \
    --text-b "..."

Example (from SemBenchmark dataset):
  python benchmarks/debug_splitter_segments.py \
    --dataset vCache/SemBenchmarkClassification \
    --row-idx 0 \
    --splitter-checkpoint /path/to/ckpt_or_dir \
    --device cuda
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from tensordict.tensordict import TensorDict

from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.MaxSimEnv import get_segments_from_token_pointers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splitter-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")

    # Option A: provide texts directly
    parser.add_argument("--text-a", default=None)
    parser.add_argument("--text-b", default=None)

    # Option B: load from dataset row
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--row-idx", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-cache-base", default=os.environ.get("HF_CACHE_BASE", "/tmp/hf"))
    args = parser.parse_args()

    # Mirror support if user has it set
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if (args.text_a is None or args.text_b is None) and args.dataset is None:
        raise SystemExit("Provide either --text-a/--text-b or --dataset (with --row-idx).")

    if args.dataset is not None and (args.text_a is None or args.text_b is None):
        cache_dir = os.path.join(args.hf_cache_base, "datasets")
        ds = load_dataset(args.dataset, split=args.split, cache_dir=cache_dir)
        row = ds[int(args.row_idx)]
        text_a = row["prompt"]
        # For debugging, compare the prompt against itself unless the dataset contains another field.
        # Many SemBenchmark datasets are single-prompt tasks; you can override with --text-b.
        text_b = row.get("prompt_b") or row.get("question_b") or row.get("prompt") or ""
    else:
        text_a = args.text_a or ""
        text_b = args.text_b or ""

    shared_embedder = EmbeddingModel(device=args.device)
    splitter = MaxSimSplitter(
        checkpoint_path=args.splitter_checkpoint,
        device=args.device,
        embedding_model=shared_embedder,
    )

    # NOTE: We intentionally do NOT call `MaxSimSplitter.debug_split_pair()` here.
    # Some deployments have a buggy debug helper. Instead, run the policy directly and
    # reconstruct segments from token pointers (same semantics as the env/policy).
    inputs = splitter.generator.tokenizer(
        [text_a, text_b],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    ).to(splitter.device)

    with torch.inference_mode():
        hs = splitter.generator.lm(**inputs).last_hidden_state  # [2, L, H]

    embeds_a = hs[0:1, :, :]
    embeds_b = hs[1:2, :, :]
    input_ids_a = inputs["input_ids"][0:1, :]
    input_ids_b = inputs["input_ids"][1:2, :]
    attention_mask_a = inputs["attention_mask"][0:1, :]
    attention_mask_b = inputs["attention_mask"][1:2, :]

    td = TensorDict(
        {
            "token_embeddings_a": embeds_a,
            "token_embeddings_b": embeds_b,
            "attention_mask_a": attention_mask_a,
            "attention_mask_b": attention_mask_b,
            "length_a": attention_mask_a.sum(dim=1),
            "length_b": attention_mask_b.sum(dim=1),
            "input_ids_a": input_ids_a,
            "input_ids_b": input_ids_b,
        },
        batch_size=1,
    )

    with torch.inference_mode():
        out = splitter.policy(
            td,
            None,
            phase="test",
            select_best=True,
            decode_type="greedy",
            compute_reward=False,
            debug=True,
        )

    actions = out["actions"][0]
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    total = int(actions.numel())
    if total % 2 != 0:
        raise ValueError(f"Expected even number of action entries (A/B interleaved), got {total}")
    max_segments = total // 2
    pointers_a = actions[0 : 2 * max_segments : 2].tolist()
    pointers_b = actions[1 : 2 * max_segments : 2].tolist()

    segments_a = get_segments_from_token_pointers(
        tokenizer=splitter.generator.tokenizer,
        input_ids=input_ids_a[0],
        attention_mask=attention_mask_a[0],
        pointers=pointers_a,
    )
    segments_b = get_segments_from_token_pointers(
        tokenizer=splitter.generator.tokenizer,
        input_ids=input_ids_b[0],
        attention_mask=attention_mask_b[0],
        pointers=pointers_b,
    )

    dbg = {
        "pointers_a": pointers_a,
        "pointers_b": pointers_b,
        "segments_a": segments_a,
        "segments_b": segments_b,
        "length_a": int(attention_mask_a[0].sum().item()),
        "length_b": int(attention_mask_b[0].sum().item()),
        "policy_info": out.get("info", {}),
    }

    print("\n=== A (text_a) ===")
    print(text_a)
    print(f"\nlength_a(tokens)={dbg['length_a']}")
    print(f"pointers_a={dbg['pointers_a']}")
    print("segments_a:")
    for i, s in enumerate(dbg["segments_a"]):
        print(f"  [{i}] {s!r}")

    print("\n=== B (text_b) ===")
    print(text_b)
    print(f"\nlength_b(tokens)={dbg['length_b']}")
    print(f"pointers_b={dbg['pointers_b']}")
    print("segments_b:")
    for i, s in enumerate(dbg["segments_b"]):
        print(f"  [{i}] {s!r}")

    info = dbg.get("policy_info", {}) or {}
    steps = info.get("debug_steps")
    if steps:
        tok = splitter.generator.tokenizer
        print("\n=== Policy per-step debug (batch[0]) ===")
        for s in steps:
            step = s.get("step")
            vca = s.get("valid_count_a")
            vcb = s.get("valid_count_b")
            pa = s.get("pointer_a")
            pb = s.get("pointer_b")
            fa = s.get("fallback_a")
            fb = s.get("fallback_b")
            print(f"- step={step} valid_a={vca} valid_b={vcb} ptr_a={pa} ptr_b={pb} fallback_a={fa} fallback_b={fb}")

            # Print top-k positions with token strings (helps diagnose punctuation mask)
            try:
                ids_a = tok.convert_ids_to_tokens(splitter.generator.tokenizer([text_a], return_tensors="pt", padding="max_length", truncation=True, max_length=512)["input_ids"][0].tolist())
                ids_b = tok.convert_ids_to_tokens(splitter.generator.tokenizer([text_b], return_tensors="pt", padding="max_length", truncation=True, max_length=512)["input_ids"][0].tolist())
                topa = s.get("topk_pos_a") or []
                topb = s.get("topk_pos_b") or []
                if topa:
                    print("  topk_a:", [(p, ids_a[p]) for p in topa[:5]])
                if topb:
                    print("  topk_b:", [(p, ids_b[p]) for p in topb[:5]])
            except Exception:
                pass


if __name__ == "__main__":
    main()


