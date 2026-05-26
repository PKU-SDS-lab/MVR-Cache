"""
Convert `eval_sembenchmark_verified.py` output JSON (args/summary/per_sample) into
benchmark4LM-compatible files:
  - results_<timestamp>.json
  - statistics_<timestamp>.json

This is useful when you already ran an evaluation and only have the saved JSON.
Latency is ignored here (filled with zeros).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _compute_statistics_json_no_latency(
    *,
    cache_hit_list: List[int],
    cache_miss_list: List[int],
    tp_list: List[int],
    fp_list: List[int],
    tn_list: List[int],
    fn_list: List[int],
) -> Dict[str, Any]:
    """
    Match the structure of `statistics_*.json` produced by benchmark4LM/splitter,
    but ignore latency (set all latency/duration values to 0 and ratios to "N/A").
    """
    n = int(len(cache_hit_list))
    hit_rate = float(sum(cache_hit_list) / n) if n > 0 else 0.0
    miss_rate = 1.0 - hit_rate
    error_rate = float(sum(fp_list) / n) if n > 0 else 0.0

    tp_sum = int(sum(tp_list))
    fp_sum = int(sum(fp_list))
    tn_sum = int(sum(tn_list))
    fn_sum = int(sum(fn_list))

    accuracy = float((tp_sum + tn_sum) / n) if n > 0 else 0.0
    precision = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    recall = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    f1_score = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # Match splitter/benchmark4LM convention: hits/misses store last-step values.
    hits_last = int(cache_hit_list[-1]) if cache_hit_list else 0
    misses_last = int(cache_miss_list[-1]) if cache_miss_list else 0

    return {
        "avg_latency": {
            "cache": {"overall": 0.0, "cache_hit": 0.0, "cache_miss": 0.0},
            "direct": {"overall": 0.0},
            "difference": {"overall": 0.0, "cache_hit": 0.0, "cache_miss": 0.0},
            "ratio": {"overall": "N/A", "cache_hit": "N/A", "cache_miss": "N/A"},
        },
        "cache": {
            "hit_rate": float(hit_rate),
            "miss_rate": float(miss_rate),
            "total_samples": int(n),
            "hits": int(hits_last),
            "misses": int(misses_last),
            "error_rate": float(error_rate),
        },
        "duration": {"vectorq": 0.0, "direct": 0.0},
        "statistics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
        },
    }


def _delta_tag(delta: float) -> str:
    """
    Stable string tag for folder names, e.g. 0.015 -> "0p015"
    """
    try:
        s = f"{float(delta):.6g}"
    except Exception:
        s = str(delta)
    return s.replace(".", "p").replace("-", "m")


def _convert_one_file(*, in_path: str, out_dir: str, ts: str) -> tuple[str, str]:
    in_path = os.path.abspath(in_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_sample = data.get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        raise ValueError(f"Input JSON missing non-empty 'per_sample' list: {in_path}")

    args_block = data.get("args", {}) if isinstance(data.get("args"), dict) else {}
    summary_block = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}

    cache_hit_list: List[int] = []
    cache_miss_list: List[int] = []
    tp_list: List[int] = []
    fp_list: List[int] = []
    tn_list: List[int] = []
    fn_list: List[int] = []

    for row in per_sample:
        if not isinstance(row, dict):
            continue
        is_hit = bool(row.get("is_hit", False))
        cache_hit_list.append(1 if is_hit else 0)
        cache_miss_list.append(0 if is_hit else 1)
        tp_list.append(int(row.get("tp", 0)))
        fp_list.append(int(row.get("fp", 0)))
        tn_list.append(int(row.get("tn", 0)))
        fn_list.append(int(row.get("fn", 0)))

    n = len(cache_hit_list)
    delta = summary_block.get("delta", args_block.get("delta", None))

    bench_results: Dict[str, Any] = {
        "config": {
            "filepath": str(args_block.get("dataset", "")),
            "embedding_model": str(args_block.get("embedding_col", "")),
            "llm_model": str(args_block.get("llm_col", "")),
            "eviction_policy": "NoEvictionPolicy()",
            "is_static_threshold": False,
            "threshold": None,
            "delta": float(delta) if delta is not None else None,
        },
        "cache_hit_list": cache_hit_list,
        "cache_miss_list": cache_miss_list,
        "tp_list": tp_list,
        "fp_list": fp_list,
        "tn_list": tn_list,
        "fn_list": fn_list,
        # latency ignored
        "latency_direct_list": [0.0] * n,
        "latency_vectorq_list": [0.0] * n,
        # not present in the source JSON -> emit empty/defaults for compatibility
        "observations_dict": {},
        "gammas_dict": {},
        "t_hats_dict": {},
        "t_primes_dict": {},
        "var_ts_dict": {},
        "global_observations_dict": {},
        "global_gamma": None,
        "global_t_hat": None,
        "global_t_prime": None,
        "global_var_t": None,
    }

    results_path = os.path.join(out_dir, f"results_{ts}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(bench_results, f, indent=4)

    stats = _compute_statistics_json_no_latency(
        cache_hit_list=cache_hit_list,
        cache_miss_list=cache_miss_list,
        tp_list=tp_list,
        fp_list=fp_list,
        tn_list=tn_list,
        fn_list=fn_list,
    )
    stats_path = os.path.join(out_dir, f"statistics_{ts}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    return results_path, stats_path


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-json", help="Path to a single results_verified_delta*.json")
    g.add_argument(
        "--input-dir",
        help="Folder containing multiple results_verified*.json files (e.g. OriginalMethodNewDatasetv3).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output directory. "
            "If --input-json, files are written directly here. "
            "If --input-dir, a subfolder per input file is created here."
        ),
    )
    p.add_argument(
        "--pattern",
        default="results_verified*.json",
        help="Glob used under --input-dir to find inputs (default: results_verified*.json).",
    )
    p.add_argument(
        "--timestamp",
        default=None,
        help="Timestamp string used in filenames. Default: YYYY-MM-DD_HH-MM",
    )
    args = p.parse_args()

    ts = str(args.timestamp) if args.timestamp else datetime.now().strftime("%Y-%m-%d_%H-%M")

    out_root = os.path.abspath(str(args.output_dir))
    os.makedirs(out_root, exist_ok=True)

    if args.input_json:
        results_path, stats_path = _convert_one_file(in_path=str(args.input_json), out_dir=out_root, ts=ts)
        print(f"Wrote: {results_path}")
        print(f"Wrote: {stats_path}")
        return

    # --input-dir mode
    in_dir = Path(str(args.input_dir)).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        raise ValueError(f"--input-dir is not a directory: {in_dir}")

    inputs = sorted(in_dir.glob(str(args.pattern)))
    if not inputs:
        raise ValueError(f"No files matched pattern '{args.pattern}' in {in_dir}")

    for pth in inputs:
        # Create deterministic per-input subfolder name.
        # Prefer delta tag if filename contains it, else use the stem.
        stem = pth.stem
        subdir = os.path.join(out_root, stem)
        try:
            results_path, stats_path = _convert_one_file(in_path=str(pth), out_dir=subdir, ts=ts)
            print(f"[OK] {pth.name} -> {subdir}")
            print(f"  - {results_path}")
            print(f"  - {stats_path}")
        except Exception as e:
            print(f"[SKIP] {pth.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

