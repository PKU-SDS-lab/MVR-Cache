"""
Plot running cache hit-rate curves from two JSON result files produced by:
  benchmarks/eval_sembenchmark_verified.py
  benchmarks/eval_sembenchmark_verified_splitter.py

Example:
  python benchmarks/plot_verified_results_compare.py \
    --base /home/ali/vcahce/results_verified.json \
    --other /home/ali/vcahce/results/verified_splitter_cuda.json \
    --out /home/ali/vcahce/results_verified_compare_hit_rate.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class LoadedResults:
    n: int
    sample_index: np.ndarray  # shape (n,) int
    running_hit_rate: np.ndarray  # shape (n,) float in [0,1]


def _running_hit_rate_from_is_hit(is_hit: np.ndarray) -> np.ndarray:
    hits = np.cumsum(is_hit.astype(np.int64))
    denom = np.arange(1, len(is_hit) + 1, dtype=np.int64)
    return hits / denom


def _load_verified_results(path: str) -> LoadedResults:
    with open(path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    per_sample = data.get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        raise ValueError(f"Expected non-empty 'per_sample' list in {path}")

    sample_index = np.array([int(r["sample_index"]) for r in per_sample], dtype=np.int64)

    if "running_hit_rate" in per_sample[0]:
        running_hit_rate = np.array(
            [float(r["running_hit_rate"]) for r in per_sample], dtype=np.float64
        )
    else:
        is_hit = np.array([bool(r["is_hit"]) for r in per_sample], dtype=bool)
        running_hit_rate = _running_hit_rate_from_is_hit(is_hit)

    if (
        sample_index.ndim != 1
        or running_hit_rate.ndim != 1
        or sample_index.shape[0] != running_hit_rate.shape[0]
    ):
        raise ValueError("Malformed per_sample arrays")

    n = int(sample_index.shape[0])
    return LoadedResults(n=n, sample_index=sample_index, running_hit_rate=running_hit_rate)


def _default_out_path(base_path: str, other_path: str) -> str:
    base = os.path.splitext(os.path.basename(base_path))[0]
    other = os.path.splitext(os.path.basename(other_path))[0]
    return os.path.join(os.path.dirname(base_path) or ".", f"{base}_vs_{other}_hit_rate.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Baseline results JSON (e.g., results_verified.json).")
    parser.add_argument("--other", required=True, help="Comparison results JSON (e.g., verified_splitter_cuda.json).")
    parser.add_argument("--out", default=None, help="Output image path (png/pdf).")
    parser.add_argument("--base-label", default="vcache", help="Legend label for baseline curve.")
    parser.add_argument("--other-label", default="vcache_maxsim", help="Legend label for comparison curve.")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Show plot window (if available).")
    args = parser.parse_args()

    # Lazy import so this file can be imported without matplotlib installed.
    import matplotlib.pyplot as plt

    base = _load_verified_results(args.base)
    other = _load_verified_results(args.other)

    if base.n != other.n:
        raise ValueError(f"Different sample counts: base={base.n}, other={other.n}")

    x = base.sample_index
    y0 = base.running_hit_rate
    y1 = other.running_hit_rate
    improvement_abs_pct = (float(y1[-1]) - float(y0[-1])) * 100.0

    out_path = args.out or _default_out_path(args.base, args.other)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, y0 * 100.0, linewidth=2.0, label=f"{args.base_label} (final={y0[-1]*100:.2f}%)")
    plt.plot(
        x,
        y1 * 100.0,
        linewidth=2.0,
        label=f"{args.other_label} (+{improvement_abs_pct:.2f}% abs, final={y1[-1]*100:.2f}%)",
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Running Cache Hit Rate (%)")
    plt.grid(True, alpha=0.3)
    title = args.title or f"Running Cache Hit Rate ({os.path.basename(args.base)})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved plot -> {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()


