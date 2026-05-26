"""
Run a single, focused configuration of `benchmarks/benchmark.py` without editing its globals.

This is useful because `benchmarks/benchmark.py` is configured as a large sweep by default.

Requested default in chat:
- Dataset: SemBenchmarkClassification
- Baseline: vCache verified (VerifiedDecisionPolicy / Baseline.VCacheLocal)
- Delta: 0.02
- Max samples: full dataset (45000)
"""

from __future__ import annotations

import argparse

import benchmarks.benchmark as bm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help="Delta for VerifiedDecisionPolicy (vCache Local).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=45000,
        help="Max samples to run for SemBenchmarkClassification (use 45000 for full train split).",
    )
    args = parser.parse_args()

    # Restrict to exactly one run-combination (SemBenchmarkClassification / GTE / LLAMA_3_8B).
    bm.RUN_COMBINATIONS = [
        (
            bm.EmbeddingModel.GTE,
            bm.LargeLanguageModel.LLAMA_3_8B,
            bm.Dataset.SEM_BENCHMARK_CLASSIFICATION,
            bm.GeneratePlotsOnly.NO,
            bm.StringComparisonSimilarityEvaluator(),
            bm.MRUEvictionPolicy(
                max_size=100000, watermark=0.99, eviction_percentage=0.1
            ),
            int(args.max_samples),
        )
    ]

    # Only run vCache verified/local baseline.
    bm.BASELINES_TO_RUN = [bm.Baseline.VCacheLocal]

    # Only one delta.
    bm.DELTAS = [float(args.delta)]

    # No static threshold sweeps.
    bm.STATIC_THRESHOLDS = []

    # Single run (no CI repeats).
    bm.CONFIDENCE_INTERVALS_ITERATIONS = 1

    bm.main()


if __name__ == "__main__":
    main()


