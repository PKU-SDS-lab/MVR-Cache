"""
Line-profiler harness for `benchmarks/eval_sembenchmark_verified_splitter.py`.

Goal: identify per-line bottlenecks in the *core vCache + VerifiedSplitter* hot path:
  - embedding computation (BGE / EmbeddingModel)
  - vector DB KNN lookup (HNSW)
  - MaxSim splitter (token embeddings + RL policy decode)
  - VerifiedSplitterDecisionPolicy similarity computation

Usage (example):
  conda activate /data1/conda_envs/RLSemanticCaching
  python benchmarks/line_profile_verified_splitter.py \
    --dataset vCache/SemBenchmarkClassification \
    --llm-col response_llama_3_8b \
    --splitter-checkpoint /data2/ali/checkpoints_words \
    --splitter-device cuda \
    --candidate-k 1 \
    --max-samples 25 \
    --profile-samples 20 \
    --warmup-samples 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

from datasets import load_dataset

from vcache.config import VCacheConfig
from vcache.main import VCache
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.vcache_core.cache.embedding_engine.strategies.bge import BGEEmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db import HNSWLibVectorDB
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import NoEvictionPolicy
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter
from vcache.vcache_core.splitter.AdaptedPointerNetworkPolicy import (
    AdaptedPointerNetworkPolicy,
    CrossAttentionBlock,
)
from vcache.vcache_core.splitter.MaxSimEnv import MaxSimEnv
from vcache.vcache_policy.strategies.verified_splitter import VerifiedSplitterDecisionPolicy


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


def _build_vcache(
    *,
    splitter_checkpoint: str,
    splitter_device: str,
    delta: float,
    candidate_k: int,
    use_cached_candidate_segments: bool = False,
) -> VCache:
    inference_engine = BenchmarkInferenceEngine()

    # Share a single EmbeddingModel across vector DB + splitter (matches eval script behavior)
    shared_embedder = EmbeddingModel(device=splitter_device)
    embedding_engine = BGEEmbeddingEngine(embedding_model=shared_embedder)

    config = VCacheConfig(
        embedding_engine=embedding_engine,
        inference_engine=inference_engine,
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=StringComparisonSimilarityEvaluator(),
        vector_db=HNSWLibVectorDB(),
    )

    splitter = MaxSimSplitter(
        checkpoint_path=splitter_checkpoint,
        device=splitter_device,
        embedding_model=shared_embedder,
    )
    policy = VerifiedSplitterDecisionPolicy(
        delta=delta,
        splitter=splitter,
        device=splitter_device,
        candidate_selection="top_k",
        candidate_k=candidate_k,
        use_cached_candidate_segments=use_cached_candidate_segments,
    )

    return VCache(config=config, policy=policy)


def _run_loop(
    *,
    vcache: VCache,
    rows: list[dict[str, Any]],
    llm_col: str,
    profile_samples: int,
    sleep_s: float,
) -> None:
    # Profile window only (warmup should be done outside the profiler)
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
    parser.add_argument("--splitter-checkpoint", required=True)
    parser.add_argument("--splitter-device", default="cuda")
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--candidate-k", type=int, default=1)
    parser.add_argument(
        "--use-cached-candidate-segments",
        action="store_true",
        help="Cache candidate MaxSim segment tensors in metadata and only segment the query at request time.",
    )
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--profile-samples", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--hf-cache-base", default=os.environ.get("HF_CACHE_BASE", None))
    parser.add_argument("--output", default="results/line_profile_verified_splitter.txt")
    args = parser.parse_args()

    # Ensure mirror (matches eval script expectations)
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
    # HF datasets slicing/indexing can return column dicts; for profiling we want a plain list of row dicts.
    rows_list: list[dict[str, Any]] = [dict(r) for r in rows]

    vcache = _build_vcache(
        splitter_checkpoint=args.splitter_checkpoint,
        splitter_device=args.splitter_device,
        delta=float(args.delta),
        candidate_k=int(args.candidate_k),
        use_cached_candidate_segments=bool(args.use_cached_candidate_segments),
    )

    # Warmup (exclude one-time allocations/compilation). This intentionally runs
    # OUTSIDE the line-profiler so per-sample averages reflect steady-state.
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

    # Attach line-profiler to the hot functions (unbound methods are fine)
    lp = LineProfiler()
    lp.add_function(VCache.infer_with_cache_info)
    lp.add_function(VerifiedSplitterDecisionPolicy.process_request)
    lp.add_function(VerifiedSplitterDecisionPolicy._select_nn_by_maxsim)
    lp.add_function(VerifiedSplitterDecisionPolicy._maxsim_similarity)
    # New optimized path (query encoded once)
    lp.add_function(VerifiedSplitterDecisionPolicy._select_nn_by_maxsim_with_query)
    lp.add_function(VerifiedSplitterDecisionPolicy._maxsim_similarity_from_encoded)

    from vcache.vcache_core.cache.cache import Cache
    from vcache.vcache_core.cache.embedding_engine.strategies.bge import BGEEmbeddingEngine
    from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
        HNSWLibVectorDB as _HNSWImpl,
    )

    lp.add_function(Cache.get_knn)
    lp.add_function(Cache.get_knn_from_embedding)
    lp.add_function(BGEEmbeddingEngine.get_embedding)
    lp.add_function(EmbeddingModel.get_embedding)
    lp.add_function(EmbeddingModel.get_embeddings_tensor)
    lp.add_function(MaxSimSplitter.split_pair_return_segments)
    lp.add_function(MaxSimSplitter.encode_text)
    lp.add_function(MaxSimSplitter.split_pair_return_maxsim_tensors_from_encoded)
    lp.add_function(_HNSWImpl.get_knn)

    # RL policy internals (this is what dominates `MaxSimSplitter.split_pair_return_segments`)
    lp.add_function(AdaptedPointerNetworkPolicy.forward)
    lp.add_function(AdaptedPointerNetworkPolicy._forward)
    lp.add_function(CrossAttentionBlock.forward)
    # Reward computation (note: MaxSimSplitter uses `out['actions']` only, but policy computes reward)
    lp.add_function(MaxSimEnv._get_reward)
    lp.add_function(MaxSimEnv.raw_score_text)

    # Run + record
    # Reset candidate-encode counters so "exclude candidate embedding time" is computed
    # strictly over the profiled window (warmup excluded).
    try:
        pol = vcache.vcache_policy
        setattr(pol, "_timing_candidate_encode_s", 0.0)
        setattr(pol, "_timing_candidate_encode_n", 0)
    except Exception:
        pass

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
        # Optional timing counters (best-effort; not part of line_profiler)
        try:
            pol = vcache.vcache_policy
            cand_s = getattr(pol, "_timing_candidate_encode_s", None)
            cand_n = getattr(pol, "_timing_candidate_encode_n", None)
            if cand_s is not None and cand_n is not None:
                f.write(
                    f"candidate_encode_total_s={cand_s}\n"
                    f"candidate_encode_calls={cand_n}\n"
                    f"candidate_encode_avg_ms={(float(cand_s) / max(1,int(cand_n))) * 1e3:.3f}\n"
                    f"candidate_encode_ms_per_profile_sample={(float(cand_s) / max(1,int(args.profile_samples))) * 1e3:.3f}\n"
                    "\n"
                )
        except Exception:
            pass
        # ms is easier to read for GPU-heavy workloads
        lp.print_stats(stream=f, output_unit=1e-3)

    print(f"[OK] Line-profiler report saved to {args.output}")

    # Clean shutdown (policy uses background threads)
    time.sleep(0.1)
    vcache.vcache_policy.shutdown()


if __name__ == "__main__":
    main()


