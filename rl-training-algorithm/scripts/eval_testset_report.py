#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the configured "test set" and print detailed metrics.

This script mirrors the test-time behavior of this repo:
- Build a generator/env from either:
  - pairs JSON (sentence_1/sentence_2/correct), or
  - parquet prompts (random pairs with labels computed by id_set/string), or
  - prompts txt (random pairs, usually no labels)
- Run the policy in eval mode (greedy by default)
- Report:
  - reward stats (mean/std/min/max)
  - BCE stats if td["correct"] is available (mean BCE, accuracy, avg prob)
  - segmentation diagnostics:
    - fallback rate per side (steps where pointer == eff_len-1)
    - average number of segments per sample per side

Examples (parquet + id_set labels):
  python scripts/eval_testset_report.py \
    --ckpt /data2/ali/checkpoints_new/epoch=4-step=1045.ckpt \
    --test_parquet /data2/ali/LMArena/train_10k.parquet \
    --parquet_text_column prompt \
    --label_mode id_set \
    --gpu 0
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from embedding_model import EmbeddingModel  # noqa: E402
from MaxSimEnv import MaxSimEnv  # noqa: E402
from MaxSimGenerator import MaxSimGenerator  # noqa: E402
from AdaptedPointerNetworkPolicy import AdaptedPointerNetworkPolicy  # noqa: E402
from RL4COTrainer import ResumeFriendlyREINFORCE, load_pairs_from_json, load_prompts_from_file  # noqa: E402


def _effective_len(td, *, ids_key: str, len_key: str, tokenizer) -> torch.Tensor:
    """Compute effective length excluding trailing [SEP]/[EOS] (matches env/policy semantics)."""
    length = td[len_key].long()
    last_idx = (length - 1).clamp(min=0)
    last_tok = td[ids_key].gather(1, last_idx.unsqueeze(1)).squeeze(1)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    is_last_special = torch.zeros_like(last_tok, dtype=torch.bool)
    if sep_id is not None:
        is_last_special |= last_tok == sep_id
    if eos_id is not None:
        is_last_special |= last_tok == eos_id
    eff_len = (length - is_last_special.long()).clamp(min=2)
    return eff_len


def _segments_count_from_pointers(pointers: torch.Tensor, eff_len: torch.Tensor) -> torch.Tensor:
    """Compute #segments per sample: (#unique valid boundaries) + 1 tail segment."""
    # pointers: [B, K] (token indices). Valid boundaries are < eff_len-1.
    B, K = pointers.shape
    end_pos = (eff_len - 1).view(B, 1)  # [B,1]
    valid = pointers < end_pos
    # Count unique valid pointers per row (small K, do per-row python for clarity)
    out = torch.zeros(B, dtype=torch.long, device=pointers.device)
    for i in range(B):
        vals = pointers[i][valid[i]].detach().to("cpu").tolist()
        out[i] = len(set(int(v) for v in vals)) + 1
    return out

def _segment_lengths_from_pointers(
    pointers: torch.Tensor, eff_len: torch.Tensor, *, skip_cls: bool = True
) -> list[list[int]]:
    """Return per-sample segment token lengths induced by pointers.

    Semantics match this repo:
    - pointer p means segment ends at p (inclusive), i.e. end = p+1
    - [CLS] token at index 0 is skipped for segment text, so first segment starts at 1
    - we ignore pointers >= eff_len-1 (fallback/end) as explicit boundaries
    """
    B, K = pointers.shape
    out: list[list[int]] = []
    for i in range(B):
        L = int(eff_len[i].item())
        end_limit = max(1, L - 1)
        ps = [int(x) for x in pointers[i].detach().to("cpu").tolist()]
        ps = [p for p in ps if 0 <= p < end_limit]  # ignore fallback/end
        bounds = sorted(set(ps))
        seg_lens: list[int] = []
        prev = 0
        for p in bounds:
            end = p + 1
            start = (prev + 1) if (skip_cls and prev > 0) else (1 if skip_cls else prev)
            if end > start:
                seg_lens.append(int(end - start))
            prev = p
        tail_start = (prev + 1) if (skip_cls and prev > 0) else (1 if skip_cls else prev)
        if tail_start < L:
            seg_lens.append(int(L - tail_start))
        out.append(seg_lens if seg_lens else [max(0, L - (1 if skip_cls else 0))])
    return out


@dataclass
class Accum:
    n: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    reward_min: float = 1e9
    reward_max: float = -1e9

    bce_sum: float = 0.0
    acc_sum: float = 0.0
    prob_sum: float = 0.0
    has_labels: bool = False

    fallback_a_sum: float = 0.0
    fallback_b_sum: float = 0.0
    steps_total: int = 0

    seg_a_sum: float = 0.0
    seg_b_sum: float = 0.0

    # Length stats
    prompt_len_a_sum: float = 0.0
    prompt_len_b_sum: float = 0.0
    prompt_len_a_min: float = 1e9
    prompt_len_a_max: float = -1e9
    prompt_len_b_min: float = 1e9
    prompt_len_b_max: float = -1e9

    seg_len_a_sum: float = 0.0
    seg_len_b_sum: float = 0.0
    seg_len_a_min: float = 1e9
    seg_len_a_max: float = -1e9
    seg_len_b_min: float = 1e9
    seg_len_b_max: float = -1e9
    seg_len_count_a: int = 0
    seg_len_count_b: int = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Lightning .ckpt path")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_segments", type=int, default=8)
    ap.add_argument("--decode_type", type=str, default="greedy", choices=["greedy", "sampling"])
    ap.add_argument("--limit_batches", type=int, default=0, help="If >0, only evaluate first N batches")
    ap.add_argument("--seed", type=int, default=44)

    # Data sources (one of these)
    ap.add_argument("--test_pairs_json", type=str, default=None)
    ap.add_argument("--test_file", type=str, default=None)
    ap.add_argument("--test_parquet", type=str, default=None)
    ap.add_argument("--parquet_text_column", type=str, default="prompt")
    ap.add_argument("--label_mode", type=str, default="none", choices=["none", "auto", "id_set", "string"])
    ap.add_argument("--response_column", type=str, default=None)

    # Policy options
    ap.add_argument("--policy_mode", type=str, default="separate", choices=["joint", "separate"])
    ap.add_argument("--split_on_space", action="store_true")
    ap.add_argument("--split_words_before", action="store_true")

    args = ap.parse_args()

    if sum(x is not None for x in [args.test_pairs_json, args.test_file, args.test_parquet]) != 1:
        raise SystemExit("Provide exactly one of --test_pairs_json, --test_file, --test_parquet")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Load data spec
    test_pairs = load_pairs_from_json(args.test_pairs_json) if args.test_pairs_json else None
    test_prompts = load_prompts_from_file(args.test_file) if args.test_file else None

    embedding_model = EmbeddingModel(device=device)

    test_generator = MaxSimGenerator(
        prompts=test_prompts,
        pairs=test_pairs,
        parquet_path=args.test_parquet if test_pairs is None and test_prompts is None else None,
        parquet_text_column=str(args.parquet_text_column),
        label_mode=str(args.label_mode),
        response_column=args.response_column,
        sampling_mode=("pairs" if test_pairs is not None else "random"),
        max_len=int(args.max_len),
        embedding_model=embedding_model,
        seed=int(args.seed),
    )
    test_env = MaxSimEnv(
        generator=test_generator,
        max_segments=int(args.max_segments),
        embedding_model=embedding_model,
        device=device,
    )

    policy = AdaptedPointerNetworkPolicy(
        test_env,
        embedding_dim=768,
        hidden_dim=768,
        max_segments=int(args.max_segments),
        policy_mode=str(args.policy_mode),
        split_on_space=bool(args.split_on_space),
        split_words_before=bool(args.split_words_before),
    )

    # Build REINFORCE module just to reuse calibrator + consistent weight loading
    model = ResumeFriendlyREINFORCE(
        env=test_env,
        policy=policy,
        baseline="rollout",
        train_data_size=1,
        val_data_size=1,
        batch_size=int(args.batch_size),
        dataloader_num_workers=0,
        optimizer_kwargs={"lr": 1e-4},
    )
    model.strict_loading = False

    print(f"[LOAD] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    model.to(device)
    model.eval()
    tok = embedding_model.tokenizer

    # Define evaluation size:
    # - pairs json: evaluate all pairs
    # - prompts/parquet: evaluate N random pairs where N = number of prompts/rows (mirrors RL4COTrainer.py)
    if test_pairs is not None:
        test_size = len(test_pairs)
    elif test_prompts is not None:
        test_size = len(test_prompts)
    else:
        import pyarrow.parquet as pq

        test_size = int(pq.ParquetFile(args.test_parquet).metadata.num_rows)

    dataset = test_env.dataset(test_size, phase="test")
    loader = DataLoader(dataset, batch_size=int(args.batch_size), num_workers=0, collate_fn=dataset.collate_fn)

    acc = Accum()
    K = int(args.max_segments)

    with torch.no_grad():
        for bi, td in enumerate(loader):
            if args.limit_batches and bi >= int(args.limit_batches):
                break
            td = td.to(device)

            out = model.policy(td, test_env, phase="test", decode_type=str(args.decode_type), debug=False)
            reward = out["reward"].view(-1).float()  # [B]
            B = int(reward.numel())

            # Reward stats
            r_cpu = reward.detach().to("cpu")
            acc.n += B
            acc.reward_sum += float(r_cpu.sum().item())
            acc.reward_sq_sum += float((r_cpu * r_cpu).sum().item())
            acc.reward_min = min(acc.reward_min, float(r_cpu.min().item()))
            acc.reward_max = max(acc.reward_max, float(r_cpu.max().item()))

            # BCE stats (if labels present)
            if hasattr(td, "keys") and ("correct" in td.keys()):
                acc.has_labels = True
                c = td["correct"].view(-1).float()
                sim = torch.clamp(reward, -1.0, 1.0)
                logits = model.calibrator.logits(sim)
                bce = F.binary_cross_entropy_with_logits(logits, c, reduction="mean")
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).float()
                batch_acc = (pred == c).float().mean()

                acc.bce_sum += float(bce.detach().cpu().item()) * B
                acc.acc_sum += float(batch_acc.detach().cpu().item()) * B
                acc.prob_sum += float(prob.detach().cpu().mean().item()) * B

            # Segmentation diagnostics: fallback + segment counts
            actions = out["actions"]
            if actions.dim() != 2 or actions.size(1) != 2 * K:
                # Support [B, K, 2]
                if actions.dim() == 3 and actions.size(-1) == 2:
                    pa = actions[:, :, 0]
                    pb = actions[:, :, 1]
                else:
                    raise RuntimeError(f"Unexpected actions shape: {tuple(actions.shape)}")
            else:
                pa = actions[:, 0::2]
                pb = actions[:, 1::2]

            eff_a = _effective_len(td, ids_key="input_ids_a", len_key="length_a", tokenizer=tok)
            eff_b = _effective_len(td, ids_key="input_ids_b", len_key="length_b", tokenizer=tok)
            end_a = (eff_a - 1).view(-1, 1)
            end_b = (eff_b - 1).view(-1, 1)
            fb_a = (pa >= end_a).float().mean().item()
            fb_b = (pb >= end_b).float().mean().item()

            acc.fallback_a_sum += fb_a * B
            acc.fallback_b_sum += fb_b * B
            acc.steps_total += B * K

            seg_a = _segments_count_from_pointers(pa, eff_a).float().mean().item()
            seg_b = _segments_count_from_pointers(pb, eff_b).float().mean().item()
            acc.seg_a_sum += seg_a * B
            acc.seg_b_sum += seg_b * B

            # Prompt length stats (token counts excluding special tail; still includes [CLS] in eff_len)
            la_cpu = eff_a.detach().to("cpu")
            lb_cpu = eff_b.detach().to("cpu")
            acc.prompt_len_a_sum += float(la_cpu.sum().item())
            acc.prompt_len_b_sum += float(lb_cpu.sum().item())
            acc.prompt_len_a_min = min(acc.prompt_len_a_min, float(la_cpu.min().item()))
            acc.prompt_len_a_max = max(acc.prompt_len_a_max, float(la_cpu.max().item()))
            acc.prompt_len_b_min = min(acc.prompt_len_b_min, float(lb_cpu.min().item()))
            acc.prompt_len_b_max = max(acc.prompt_len_b_max, float(lb_cpu.max().item()))

            # Segment length stats
            seg_lens_a = _segment_lengths_from_pointers(pa, eff_a, skip_cls=True)
            seg_lens_b = _segment_lengths_from_pointers(pb, eff_b, skip_cls=True)
            for i in range(B):
                for Ls in seg_lens_a[i]:
                    acc.seg_len_a_sum += float(Ls)
                    acc.seg_len_count_a += 1
                    acc.seg_len_a_min = min(acc.seg_len_a_min, float(Ls))
                    acc.seg_len_a_max = max(acc.seg_len_a_max, float(Ls))
                for Ls in seg_lens_b[i]:
                    acc.seg_len_b_sum += float(Ls)
                    acc.seg_len_count_b += 1
                    acc.seg_len_b_min = min(acc.seg_len_b_min, float(Ls))
                    acc.seg_len_b_max = max(acc.seg_len_b_max, float(Ls))
                    acc.seg_len_b_min = min(acc.seg_len_b_min, float(Ls))
                    acc.seg_len_b_max = max(acc.seg_len_b_max, float(Ls))

    if acc.n == 0:
        raise SystemExit("No samples evaluated")

    mean = acc.reward_sum / acc.n
    var = max(0.0, acc.reward_sq_sum / acc.n - mean * mean)
    std = var ** 0.5

    print("\n=== EVAL REPORT ===")
    print(f"samples={acc.n} batch_size={args.batch_size} decode_type={args.decode_type}")
    print(f"reward: mean={mean:.6f} std={std:.6f} min={acc.reward_min:.6f} max={acc.reward_max:.6f}")

    if acc.has_labels:
        print(f"bce:   mean={acc.bce_sum / acc.n:.6f}")
        print(f"acc@0.5: {acc.acc_sum / acc.n:.4f}")
        print(f"avg_prob: {acc.prob_sum / acc.n:.4f}")
        try:
            print(f"calibrator: t={float(model.calibrator.t.detach().cpu().item()):.6f} gamma={float(torch.exp(model.calibrator.log_gamma.detach()).cpu().item()):.6f}")
        except Exception:
            pass
    else:
        print("bce:   (no labels found in td['correct'])")

    print(f"fallback_rate: A={acc.fallback_a_sum / acc.n:.4f} B={acc.fallback_b_sum / acc.n:.4f}  (fraction of steps using fallback/end)")
    print(f"avg_segments:  A={acc.seg_a_sum / acc.n:.3f} B={acc.seg_b_sum / acc.n:.3f}")

    print("\n-- Length stats (token-based) --")
    print(
        f"prompt_len eff_len: A mean={acc.prompt_len_a_sum / acc.n:.2f} min={acc.prompt_len_a_min:.0f} max={acc.prompt_len_a_max:.0f} | "
        f"B mean={acc.prompt_len_b_sum / acc.n:.2f} min={acc.prompt_len_b_min:.0f} max={acc.prompt_len_b_max:.0f}"
    )
    if acc.seg_len_count_a > 0 and acc.seg_len_count_b > 0:
        print(
            f"segment_len (excluding [CLS]): A mean={acc.seg_len_a_sum / acc.seg_len_count_a:.2f} min={acc.seg_len_a_min:.0f} max={acc.seg_len_a_max:.0f} | "
            f"B mean={acc.seg_len_b_sum / acc.seg_len_count_b:.2f} min={acc.seg_len_b_min:.0f} max={acc.seg_len_b_max:.0f}"
        )
    else:
        print("segment_len: (not available)")


if __name__ == "__main__":
    main()

