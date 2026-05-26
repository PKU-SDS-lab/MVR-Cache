#!/usr/bin/env python3
"""
Plot running cache hit rate from vCache benchmark result JSONs.

Example:
  python benchmarks/plot_running_hit_rate.py \
    --a /home/ali/vcahce/results_verified.json --a-label vcache \
    --b /home/ali/vcahce/results/verified_splitter_cuda_cachedcandsegments.json --b-label vcache_maxsim \
    --out /home/ali/vcahce/results/running_cache_hit_rate.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _extract_series(data: Dict[str, Any]) -> Tuple[List[int], List[float], float]:
    per = data.get("per_sample") or []
    xs: List[int] = []
    ys: List[float] = []
    for r in per:
        try:
            xs.append(int(r.get("sample_index")))
            ys.append(float(r.get("running_hit_rate")) * 100.0)
        except Exception:
            continue
    final = float(data.get("summary", {}).get("hit_rate", (ys[-1] / 100.0 if ys else 0.0))) * 100.0
    return xs, ys, final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="Path to baseline results JSON")
    p.add_argument("--a-label", default="a", help="Legend label for baseline")
    p.add_argument("--b", required=True, help="Path to comparison results JSON")
    p.add_argument("--b-label", default="b", help="Legend label for comparison")
    p.add_argument("--title", default=None, help="Plot title (default: Running Cache Hit Rate)")
    p.add_argument("--out", required=True, help="Output PNG path")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Install it in your env.") from e

    da = _load(args.a)
    db = _load(args.b)

    xa, ya, fa = _extract_series(da)
    xb, yb, fb = _extract_series(db)

    delta_abs = fb - fa
    title = args.title or "Running Cache Hit Rate"

    plt.figure(figsize=(12, 5))
    plt.plot(xa, ya, linewidth=2.0, label=f"{args.a_label} (final={fa:.2f}%)")
    plt.plot(xb, yb, linewidth=2.0, label=f"{args.b_label} ({delta_abs:+.2f}% abs, final={fb:.2f}%)")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Running Cache Hit Rate (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=200)

    print(f"[OK] Wrote plot to {args.out}")


if __name__ == "__main__":
    main()


