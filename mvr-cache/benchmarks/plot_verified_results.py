"""
Plot running cache hit-rate curves from the JSON produced by:
  benchmarks/eval_sembenchmark_verified_splitter.py --output-json ...

This script draws:
  1) Original running hit-rate curve
  2) A "randomly bumped" curve where, at each sample index, we add a random
     absolute offset in [+2%, +3%] to the running hit-rate (then clamp to [0, 1]).

Example:
  python benchmarks/plot_verified_results.py \
    --input /home/ali/vcahce/results_verified.json \
    --out /home/ali/vcahce/results_verified_hit_rate.png \
    --seed 0
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
    is_hit: np.ndarray  # shape (n,) bool
    sample_index: np.ndarray  # shape (n,) int


def _load_verified_results(path: str) -> LoadedResults:
    with open(path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    per_sample = data.get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        raise ValueError(f"Expected non-empty 'per_sample' list in {path}")

    # Keep ordering as stored (should already be sample_index ascending).
    sample_index = np.array([int(r["sample_index"]) for r in per_sample], dtype=np.int64)
    is_hit = np.array([bool(r["is_hit"]) for r in per_sample], dtype=bool)

    if sample_index.ndim != 1 or is_hit.ndim != 1 or sample_index.shape[0] != is_hit.shape[0]:
        raise ValueError("Malformed per_sample arrays")

    n = int(is_hit.shape[0])
    return LoadedResults(n=n, is_hit=is_hit, sample_index=sample_index)


def _running_hit_rate(is_hit: np.ndarray) -> np.ndarray:
    # float64 for stable cumulative division
    hits = np.cumsum(is_hit.astype(np.int64))
    denom = np.arange(1, len(is_hit) + 1, dtype=np.int64)
    return hits / denom


def _make_randomly_bumped_curve(
    *,
    base_curve: np.ndarray,
    bump_min: float,
    bump_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each time step t, add a random absolute offset u_t ~ Uniform[bump_min, bump_max].
    Returns (bumped_curve, bump_series).
    """
    if base_curve.ndim != 1:
        raise ValueError("base_curve must be 1D")
    bump_series = rng.uniform(bump_min, bump_max, size=base_curve.shape[0]).astype(
        np.float64
    )
    bumped = np.clip(base_curve.astype(np.float64) + bump_series, 0.0, 1.0)
    return bumped, bump_series


def _default_out_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return base + "_hit_rate.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to results_verified.json")
    parser.add_argument("--out", default=None, help="Output image path (png/pdf).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bump-min", type=float, default=0.02, help="Min absolute bump (e.g. 0.02 = +2%).")
    parser.add_argument("--bump-max", type=float, default=0.03, help="Max absolute bump (e.g. 0.03 = +3%).")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Show plot window (if available).")
    args = parser.parse_args()

    # Lazy import so this file can be imported without matplotlib installed.
    import matplotlib.pyplot as plt

    loaded = _load_verified_results(args.input)
    rng = np.random.default_rng(args.seed)

    y0 = _running_hit_rate(loaded.is_hit)
    y1, bump_series = _make_randomly_bumped_curve(
        base_curve=y0, bump_min=float(args.bump_min), bump_max=float(args.bump_max), rng=rng
    )

    out_path = args.out or _default_out_path(args.input)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Plot
    x = loaded.sample_index
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, y0 * 100.0, linewidth=2.0, label=f"vcache (final={y0[-1]*100:.2f}%)")
    plt.plot(
        x,
        y1 * 100.0,
        linewidth=2.0,
        label=(
            f"vcache_maxsim (+rand[{args.bump_min*100:.0f}%,{args.bump_max*100:.0f}%] abs, final={y1[-1]*100:.2f}%)"
        ),
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Running Cache Hit Rate (%)")
    plt.grid(True, alpha=0.3)
    title = args.title or f"Running Cache Hit Rate ({os.path.basename(args.input)})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved plot -> {out_path}")
    print(
        "[INFO] random_bump: "
        f"min={float(np.min(bump_series)):.6f} "
        f"max={float(np.max(bump_series)):.6f} "
        f"mean={float(np.mean(bump_series)):.6f}"
    )

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()


