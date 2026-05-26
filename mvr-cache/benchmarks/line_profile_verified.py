"""
Line-profiler harness for `benchmarks/eval_sembenchmark_verified.py`.

This profiles the *steady-state* per-sample costs of the VerifiedDecisionPolicy
pipeline (no splitter / MaxSim).

Key goals:
  - identify time spent in live embeddings (EmbeddingModel / BGEEmbeddingEngine)
  - identify vector DB lookup costs (HNSW)
  - identify policy decision costs (VerifiedDecisionPolicy / _Algorithm)

Notes:
  - Warmup is intentionally run OUTSIDE the line-profiler so reported per-hit
    times better reflect steady-state.
  - The policy performs background updates (thread pool + callback queue). Work
    done on those threads is *not* attributed to the main-thread line profile.
    This script focuses on main-thread latency per request; see `--sleep` to
    emulate the eval script pacing that gives background updates time to run.

Example:
  python benchmarks/line_profile_verified.py \
    --dataset vCache/SemBenchmarkClassification \
    --llm-col response_llama_3_8b \
    --device cuda \
    --delta 0.02 \
    --max-samples 30 \
    --warmup-samples 3 \
    --profile-samples 10 \
    --sleep 0.0 \
    --hf-cache-base /data2/ali/hf \
    --output results/line_profile_verified_cuda_clean.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

from datasets import load_dataset

from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.bge import BGEEmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import NoEvictionPolicy
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_policy.strategies.verified import VerifiedDecisionPolicy, _Algorithm


def _ensure_hf_cache_env(hf_cache_base: str | None = None) -> Dict[str, str]:
    hf_cache_base = hf_cache_base or os.environ.get("HF_CACHE_BASE", "/tmp/hf")
    hf_home = os.path.join(hf_cache_base, "home")
    hf_hub_cache = os.path.join(hf_cache_base, "hub")
    hf_datasets_cache = os.path.join(hf_cache_base, "datasets")
    hf_transformers_cache = os.path.join(hf_cache_base, "transformers")

    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(hf_datasets_cache, exist_ok=True)
    os.makedirs(hf_transformers_cache, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("DATASETS_CACHE", hf_datasets_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_transformers_cache)

    return {
        "HF_CACHE_BASE": hf_cache_base,
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub_cache,
        "DATASETS_CACHE": hf_datasets_cache,
        "TRANSFORMERS_CACHE": hf_transformers_cache,
    }


def _get_id_set(row: Dict[str, Any]) -> int:
    v = row.get("id_set", -1)
    if v == -1:
        v = row.get("ID_Set", -1)
    try:
        return int(v)
    except Exception:
        return -1


def _build_vcache(*, device: str, delta: float, max_capacity: int) -> VCache:
    inference_engine = BenchmarkInferenceEngine()

    shared_embedder = EmbeddingModel(device=device)
    embedding_engine = BGEEmbeddingEngine(embedding_model=shared_embedder)

    config = VCacheConfig(
        inference_engine=inference_engine,
        embedding_engine=embedding_engine,
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=int(max_capacity),
        ),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        # For profiling we don't want expensive evaluation; string compare is cheap.
        similarity_evaluator=StringComparisonSimilarityEvaluator(),
    )

    policy = VerifiedDecisionPolicy(delta=float(delta))
    return VCache(config=config, policy=policy)


def _run_loop(
    *,
    vcache: VCache,
    rows: list[dict[str, Any]],
    llm_col: str,
    profile_samples: int,
    sleep_s: float,
) -> None:
    m = 0
    for r in rows:
        prompt = r["prompt"]
        system_prompt = r.get("output_format", "") or ""
        id_set = _get_id_set(r)
        label_response = r[llm_col]
        vcache.vcache_policy.inference_engine.set_next_response(label_response)
        vcache.infer_with_cache_info(prompt=prompt, system_prompt=system_prompt, id_set=id_set)
        m += 1
        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))
        if m >= profile_samples:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--llm-col", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--max-capacity", type=int, default=200_000)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--profile-samples", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--hf-cache-base", default=os.environ.get("HF_CACHE_BASE", None))
    parser.add_argument("--output", default="results/line_profile_verified.txt")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cache_paths = _ensure_hf_cache_env(args.hf_cache_base)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    split = f"train[:{int(args.max_samples)}]"
    rows = load_dataset(
        args.dataset,
        split=split,
        cache_dir=cache_paths["DATASETS_CACHE"],
        token=hf_token,
    )
    rows_list: list[dict[str, Any]] = [dict(r) for r in rows]

    vcache = _build_vcache(device=args.device, delta=float(args.delta), max_capacity=int(args.max_capacity))

    # Warmup outside profiler.
    warm_n = 0
    for r in rows_list:
        prompt = r["prompt"]
        system_prompt = r.get("output_format", "") or ""
        id_set = _get_id_set(r)
        label_response = r[args.llm_col]
        vcache.vcache_policy.inference_engine.set_next_response(label_response)
        vcache.infer_with_cache_info(prompt=prompt, system_prompt=system_prompt, id_set=id_set)
        warm_n += 1
        if warm_n >= int(args.warmup_samples):
            break

    try:
        from line_profiler import LineProfiler
    except Exception as e:
        raise RuntimeError(
            "line_profiler is required. Install it in the env, e.g. `pip install line_profiler`."
        ) from e

    from vcache.vcache_core.cache.cache import Cache
    from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
        HNSWLibVectorDB as _HNSWImpl,
    )

    lp = LineProfiler()
    lp.add_function(VCache.infer_with_cache_info)
    lp.add_function(VerifiedDecisionPolicy.process_request)
    lp.add_function(Cache.get_knn)
    lp.add_function(BGEEmbeddingEngine.get_embedding)
    lp.add_function(EmbeddingModel.get_embedding)
    lp.add_function(_HNSWImpl.get_knn)

    # Bayesian decision internals (can dominate once cache starts exploiting).
    lp.add_function(_Algorithm.select_action)
    lp.add_function(_Algorithm._estimate_parameters)
    lp.add_function(_Algorithm._get_tau)
    lp.add_function(_Algorithm._get_var_t)

    profiled = lp(
        lambda: _run_loop(
            vcache=vcache,
            rows=rows_list[int(args.warmup_samples) :],
            llm_col=args.llm_col,
            profile_samples=int(args.profile_samples),
            sleep_s=float(args.sleep),
        )
    )
    profiled()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(
            f"args={vars(args)}\n"
            f"python={sys.version}\n"
            f"cwd={os.getcwd()}\n"
            "\n"
        )
        lp.print_stats(stream=f, output_unit=1e-3)  # ms

    print(f"[OK] Line-profiler report saved to {args.output}")

    # Give background threads a moment, then shutdown cleanly.
    time.sleep(0.1)
    vcache.vcache_policy.shutdown()


if __name__ == "__main__":
    main()


