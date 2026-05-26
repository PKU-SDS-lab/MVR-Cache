from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List

import pandas as pd
import torch
from datasets import DownloadConfig, load_dataset
from tensordict.tensordict import TensorDict
from tqdm import tqdm


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _purge_nonlocal_vcache_modules() -> None:
    local_prefix = os.path.abspath(_PROJECT_ROOT) + os.sep
    for module_name, module in list(sys.modules.items()):
        if module_name != "vcache" and not module_name.startswith("vcache."):
            continue
        module_file = getattr(module, "__file__", None)
        module_path = None
        if module_file:
            module_path = os.path.abspath(module_file)
        else:
            module_search_paths = list(getattr(module, "__path__", []) or [])
            if module_search_paths:
                module_path = os.path.abspath(str(module_search_paths[0]))
        if module_path and not module_path.startswith(local_prefix):
            del sys.modules[module_name]


_purge_nonlocal_vcache_modules()

from vcache.vcache_core.splitter.MaxSimEnv import get_segments_from_token_pointers  # noqa: E402
from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter  # noqa: E402
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel  # noqa: E402

if "max_segments" not in MaxSimSplitter.__init__.__code__.co_varnames:
    _purge_nonlocal_vcache_modules()
    MaxSimSplitter = importlib.import_module(
        "vcache.vcache_core.splitter.MaxSimSplitter"
    ).MaxSimSplitter


def _ensure_hf_cache_env(hf_cache_base: str | None) -> Dict[str, str]:
    if not hf_cache_base:
        return {
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
            "HUB_CACHE": os.environ.get("HUGGINGFACEHUB_CACHE", ""),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
        }

    hf_home = os.path.abspath(hf_cache_base)
    hf_datasets_cache = os.path.join(hf_home, "datasets")
    hf_hub_cache = os.path.join(hf_home, "hub")
    hf_transformers_cache = os.path.join(hf_home, "transformers")
    os.makedirs(hf_datasets_cache, exist_ok=True)
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(hf_transformers_cache, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache
    os.environ["HUGGINGFACEHUB_CACHE"] = hf_hub_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_transformers_cache
    return {
        "HF_HOME": hf_home,
        "DATASETS_CACHE": hf_datasets_cache,
        "HUB_CACHE": hf_hub_cache,
        "TRANSFORMERS_CACHE": hf_transformers_cache,
    }


def _load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    dataset_is_local_file = False
    try:
        dataset_is_local_file = os.path.exists(str(args.dataset))
    except Exception:
        dataset_is_local_file = False
    if str(args.dataset).endswith(".csv") or str(args.dataset).endswith(".parquet"):
        dataset_is_local_file = True

    if dataset_is_local_file:
        dataset_path = os.path.abspath(str(args.dataset))
        if dataset_path.endswith(".csv"):
            try:
                df = pd.read_csv(dataset_path)
            except Exception:
                df = pd.read_csv(
                    dataset_path,
                    engine="python",
                    on_bad_lines="skip",
                    low_memory=False,
                )
        elif dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(
                f"Unsupported local dataset file format: {dataset_path} (expected .csv or .parquet)"
            )

        if args.prompt_col not in df.columns:
            raise ValueError(
                f"Dataset is missing prompt column '{args.prompt_col}'. Available columns: {list(df.columns)}"
            )

        rows = df.to_dict("records")
        if args.max_samples is not None:
            rows = rows[: int(args.max_samples)]
        return rows

    cache_paths = _ensure_hf_cache_env(args.hf_cache_base)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    split = "train"
    if args.max_samples is not None:
        split = f"train[:{args.max_samples}]"

    dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
    dataset_rows = load_dataset(
        args.dataset,
        split=split,
        cache_dir=cache_paths["DATASETS_CACHE"],
        token=hf_token,
        download_config=dl_cfg,
    )
    return list(dataset_rows)


def _chunk_text_by_tokens(text: str, tokenizer, chunk_token_limit: int) -> List[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    limit = max(8, int(chunk_token_limit))
    chunks: List[str] = []
    for start in range(0, len(token_ids), limit):
        chunk_ids = token_ids[start : start + limit]
        if not chunk_ids:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def _segment_single_text(splitter: MaxSimSplitter, text: str) -> List[str]:
    inputs = splitter.generator.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(splitter.device)

    with torch.inference_mode():
        hidden_states = splitter.generator.lm(**inputs).last_hidden_state

    token_emb = hidden_states[0]
    attention_mask = inputs["attention_mask"][0]
    input_ids = inputs["input_ids"][0]
    length = int(attention_mask.sum().item())

    td = TensorDict(
        {
            "token_embeddings": token_emb.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "input_ids": input_ids.unsqueeze(0),
            "length": torch.tensor([length], device=splitter.device),
        },
        batch_size=1,
    )

    with torch.inference_mode():
        out = splitter.policy.forward_single(td, decode_type="greedy", compute_reward=False)

    actions = out["actions"][0]
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions, device=splitter.device)
    pointers = actions.tolist()

    segments = get_segments_from_token_pointers(
        tokenizer=splitter.generator.tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pointers=pointers,
    )
    return [str(segment).strip() for segment in segments if str(segment).strip()]


def _segment_long_text(
    splitter: MaxSimSplitter,
    text: str,
    chunk_token_limit: int,
) -> tuple[List[str], int]:
    chunk_texts = _chunk_text_by_tokens(
        text=text,
        tokenizer=splitter.generator.tokenizer,
        chunk_token_limit=chunk_token_limit,
    )
    all_segments: List[str] = []
    for chunk_text in chunk_texts:
        all_segments.extend(_segment_single_text(splitter, chunk_text))
    return all_segments, len(chunk_texts)


def _percentile(sorted_values: List[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * float(p)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _summarize_numeric(values: List[int]) -> Dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p25": _percentile(sorted_values, 0.25),
        "p75": _percentile(sorted_values, 0.75),
        "p90": _percentile(sorted_values, 0.90),
        "p95": _percentile(sorted_values, 0.95),
        "p99": _percentile(sorted_values, 0.99),
    }


def _resolve_output_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    abs_path = os.path.abspath(str(path_value))
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return abs_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset id.")
    parser.add_argument("--prompt-col", default="prompt", help="Prompt column name.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--chunk-token-limit", type=int, default=448)
    parser.add_argument("--overlap-tokens", type=int, default=0)
    parser.add_argument("--include-full-embedding", action="store_true")
    parser.add_argument("--hf-cache-base", default=os.environ.get("HF_CACHE_BASE", None))
    parser.add_argument("--output-json", default=None, help="Optional JSON summary path.")
    parser.add_argument(
        "--save-per-sample",
        default=None,
        help="Optional JSONL path with per-sample segment statistics.",
    )
    args = parser.parse_args()

    rows = _load_rows(args)

    shared_embedder = EmbeddingModel(device=args.device)
    splitter = MaxSimSplitter(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        embedding_model=shared_embedder,
        max_segments=int(args.max_segments),
        overlap_tokens=int(args.overlap_tokens),
        include_full_embedding=bool(args.include_full_embedding),
    )

    per_sample_output_path = _resolve_output_path(args.save_per_sample)
    per_sample_f = (
        open(per_sample_output_path, "w", encoding="utf-8")
        if per_sample_output_path is not None
        else None
    )

    segment_counts: List[int] = []
    chunk_counts: List[int] = []
    prompt_char_lengths: List[int] = []
    segment_char_sums: List[int] = []

    t0 = os.times().elapsed
    pbar = tqdm(rows, desc="Computing segmentation stats", unit="samples")
    try:
        for sample_index, row in enumerate(pbar, start=1):
            prompt = str(row.get(args.prompt_col, "") or "")
            segments, n_chunks = _segment_long_text(
                splitter=splitter,
                text=prompt,
                chunk_token_limit=int(args.chunk_token_limit),
            )

            seg_count = int(len(segments))
            chunk_count = int(n_chunks)
            prompt_chars = int(len(prompt))
            seg_char_sum = int(sum(len(seg) for seg in segments))

            segment_counts.append(seg_count)
            chunk_counts.append(chunk_count)
            prompt_char_lengths.append(prompt_chars)
            segment_char_sums.append(seg_char_sum)

            if per_sample_f is not None:
                per_sample_f.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "prompt_chars": prompt_chars,
                            "chunk_count": chunk_count,
                            "segment_count": seg_count,
                            "segment_char_sum": seg_char_sum,
                        }
                    )
                    + "\n"
                )
    finally:
        if per_sample_f is not None:
            per_sample_f.close()

    elapsed = max(0.0, os.times().elapsed - t0)
    n = len(segment_counts)
    segment_counter = Counter(segment_counts)
    chunk_counter = Counter(chunk_counts)

    summary = {
        "dataset": str(args.dataset),
        "prompt_col": str(args.prompt_col),
        "n": int(n),
        "checkpoint_path": str(args.checkpoint_path),
        "device": str(args.device),
        "max_segments": int(args.max_segments),
        "chunk_token_limit": int(args.chunk_token_limit),
        "elapsed_s": float(elapsed),
        "segment_count_stats": _summarize_numeric(segment_counts),
        "chunk_count_stats": _summarize_numeric(chunk_counts),
        "prompt_char_length_stats": _summarize_numeric(prompt_char_lengths),
        "segment_char_sum_stats": _summarize_numeric(segment_char_sums),
        "segment_count_histogram": {
            str(k): int(v) for k, v in sorted(segment_counter.items(), key=lambda item: item[0])
        },
        "chunk_count_histogram": {
            str(k): int(v) for k, v in sorted(chunk_counter.items(), key=lambda item: item[0])
        },
    }

    print(f"dataset={args.dataset}")
    print(f"n={n} elapsed_s={elapsed:.2f}")
    print(
        "segment_count_stats="
        f"min:{summary['segment_count_stats']['min']:.0f} "
        f"max:{summary['segment_count_stats']['max']:.0f} "
        f"mean:{summary['segment_count_stats']['mean']:.3f} "
        f"median:{summary['segment_count_stats']['median']:.3f} "
        f"p90:{summary['segment_count_stats']['p90']:.3f} "
        f"p95:{summary['segment_count_stats']['p95']:.3f}"
    )
    print(
        "chunk_count_stats="
        f"min:{summary['chunk_count_stats']['min']:.0f} "
        f"max:{summary['chunk_count_stats']['max']:.0f} "
        f"mean:{summary['chunk_count_stats']['mean']:.3f}"
    )

    output_path = _resolve_output_path(args.output_json)
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "summary": summary}, f, indent=2)
        print(f"Results saved to {output_path}")
    if per_sample_output_path is not None:
        print(f"Per-sample stats saved to {per_sample_output_path}")


if __name__ == "__main__":
    main()
