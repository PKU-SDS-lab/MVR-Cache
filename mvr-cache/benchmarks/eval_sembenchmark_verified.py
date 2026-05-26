"""
Offline evaluation on SemBenchmark datasets using VerifiedDecisionPolicy.

This script uses live BGE embeddings (computed via EmbeddingModel) to match the
model used by the Splitter approach, ensuring a fair comparison between the two.
"""

from __future__ import annotations
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated.*",
    category=FutureWarning,
)
import argparse
from datetime import datetime
import json
import os
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm

from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.bge import (
    BGEEmbeddingEngine,
)
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import NoEvictionPolicy
from vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison import (
    BenchmarkComparisonSimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_policy.strategies.verified import VerifiedDecisionPolicy


def _ensure_hf_cache_env() -> Dict[str, str]:
    """
    Set HF cache env vars if HF_CACHE_BASE is provided (or use /tmp).
    Returns resolved paths.
    """
    hf_cache_base = os.environ.get("HF_CACHE_BASE", "/tmp/hf")
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
    # Fallback for custom datasets: allow using `label_id` as the benchmark id_set label.
    if v == -1:
        v = row.get("label_id", -1)
    try:
        return int(v)
    except Exception:
        return -1


def _score_step(
    *,
    is_cache_hit: bool,
    label_id_set: int,
    label_response: str,
    cache_response: str,
    response_metadata,
    nn_metadata,
    use_llm_judge: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Match `Benchmark.update_stats` semantics for (tp, fp, tn, fn).
    """
    if is_cache_hit:
        if label_id_set != -1:
            cache_response_correct = label_id_set == getattr(response_metadata, "id_set", -999999)
        elif use_llm_judge:
            # Not used in this script; kept for parity.
            cache_response_correct = label_response == cache_response
        else:
            cache_response_correct = answers_have_same_meaning_static(label_response, cache_response)

        if cache_response_correct:
            return 1, 0, 0, 0
        return 0, 1, 0, 0

    # cache miss
    if label_id_set != -1:
        nn_response_correct = label_id_set == getattr(nn_metadata, "id_set", -999999)
    elif use_llm_judge:
        nn_response_correct = label_response == getattr(nn_metadata, "response", "")
    else:
        nn_response_correct = answers_have_same_meaning_static(
            label_response, getattr(nn_metadata, "response", "")
        )

    if nn_response_correct:
        return 0, 0, 0, 1  # FN
    return 0, 0, 1, 0  # TN


def _delta_tag(delta: float) -> str:
    """
    Stable string tag for file names, e.g. 0.01 -> "0p01"
    """
    try:
        s = f"{float(delta):.6g}"
    except Exception:
        s = str(delta)
    return s.replace(".", "p").replace("-", "m")


def _path_with_delta_suffix(path: str, *, delta: float) -> str:
    """
    Insert `_delta{tag}` before the file extension.
    If no extension exists, append it.
    """
    tag = _delta_tag(delta)
    base, ext = os.path.splitext(path)
    if ext:
        return f"{base}_delta{tag}{ext}"
    return f"{path}_delta{tag}"

class TimingCollector:
    """
    Low-overhead accumulator for embedding/retrieval wall times inside VerifiedDecisionPolicy.
    """

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, name: str, dt_seconds: float) -> None:
        k = str(name)
        v = float(dt_seconds)
        self.sums[k] = self.sums.get(k, 0.0) + v
        self.counts[k] = self.counts.get(k, 0) + 1

    def get(self, name: str) -> float:
        return float(self.sums.get(str(name), 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset id (e.g. vCache/SemBenchmarkClassification) OR a local .csv/.parquet file path.",
    )
    parser.add_argument(
        "--embedding-col",
        help="Embedding column (optional, no longer used for embeddings as BGE is live).",
    )
    parser.add_argument(
        "--llm-col",
        required=False,
        default=None,
        help=(
            "LLM response column, e.g. response_llama_3_8b. "
            "Optional when --similarity-evaluator=benchmark_id_set (label-only evaluation)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate. If None, use all.",
    )
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=None,
        help="If set, evaluate multiple deltas in one run (e.g. --deltas 0.01 0.02 0.03). Overrides --delta.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for live BGE embedding calculation (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.002,
        help="Per-step sleep to allow VerifiedDecisionPolicy background updates (match benchmark.py).",
    )
    parser.add_argument(
        "--similarity-evaluator",
        choices=["benchmark_id_set", "string"],
        default="benchmark_id_set",
        help="How to evaluate correctness (match benchmark.py run-combination).",
    )
    parser.add_argument("--max-capacity", type=int, default=200_000)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save per-sample results for curve plotting.",
    )
    parser.add_argument(
        "--aggregate-json",
        type=str,
        default=None,
        help=(
            "If set AND you run with --deltas, write an aggregate summary JSON here. "
            "By default (aggregate-json unset), multi-delta runs only write per-delta files."
        ),
    )
    parser.add_argument(
        "--save-cache-hit-samples",
        type=str,
        default=None,
        help="If set, write ALL cache-hit samples to this path as JSONL (one record per hit).",
    )
    parser.add_argument(
        "--timing-output",
        type=str,
        default=None,
        help=(
            "If set, append timing summary for each delta to this text file. "
            "Fields: embedding_s, retrieval_s, total_s. Times are totals over n samples."
        ),
    )
    args = parser.parse_args()

    # Mirror: must be set before importing HF libs in a fresh process.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # Avoid flaky XetHub-backed downloads (cas-bridge.xethub.hf.co) and use standard HTTP downloads.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Be more tolerant of slow connections for large parquet files.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    # Load dataset: support both HF dataset IDs and local CSV/parquet files.
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
                # More tolerant CSV parsing for large / messy files
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

        if "prompt" not in df.columns:
            raise ValueError(
                f"Local dataset is missing required column 'prompt'. Available columns: {list(df.columns)}"
            )

        # If user asked for benchmark_id_set but the file doesn't have usable ids, fall back to string.
        id_col = None
        if "ID_Set" in df.columns:
            id_col = "ID_Set"
        elif "id_set" in df.columns:
            id_col = "id_set"
        elif "label_id" in df.columns:
            id_col = "label_id"
        if args.similarity_evaluator == "benchmark_id_set":
            has_usable_ids = False
            if id_col is not None:
                try:
                    s = pd.to_numeric(df[id_col], errors="coerce").fillna(-1)
                    has_usable_ids = bool((s.astype(int) != -1).any())
                except Exception:
                    has_usable_ids = False
            if not has_usable_ids:
                args.similarity_evaluator = "string"
        # If we fell back to string evaluation, we MUST have a response column.
        if args.similarity_evaluator == "string":
            if args.llm_col is None or str(args.llm_col) == "":
                raise ValueError(
                    "similarity-evaluator='string' requires --llm-col, but none was provided. "
                    f"Available columns: {list(df.columns)}"
                )
            if args.llm_col not in df.columns:
                raise ValueError(
                    f"Local dataset is missing required LLM response column '{args.llm_col}'. Available columns: {list(df.columns)}"
                )

        rows = df.to_dict("records")
        if args.max_samples is not None:
            rows = rows[: int(args.max_samples)]
    else:
        cache_paths = _ensure_hf_cache_env()
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        split = "train"
        if args.max_samples is not None:
            split = f"train[:{args.max_samples}]"

        dl_cfg = DownloadConfig(
            resume_download=True,
            max_retries=50,
        )
        rows = load_dataset(
            args.dataset,
            split=split,
            cache_dir=cache_paths["DATASETS_CACHE"],
            token=hf_token,
            download_config=dl_cfg,
        )

    # Decide deltas to run
    deltas_to_run: List[float]
    if args.deltas is not None and len(args.deltas) > 0:
        deltas_to_run = [float(d) for d in args.deltas]
    else:
        deltas_to_run = [float(args.delta)]

    benchmark_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    def _resolve_benchmark_output_dir(*, delta: float) -> str:
        # Default location: <project_root>/results/benchmark_compat/vcache_local_<delta>_run_1/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        base_dir = os.path.join(project_root, "results", "benchmark_compat")
        os.makedirs(base_dir, exist_ok=True)
        out_dir = os.path.join(base_dir, f"vcache_local_{delta}_run_1")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _compute_statistics_json(
        *,
        cache_hit_list: list[int],
        cache_miss_list: list[int],
        tp_list: list[int],
        fp_list: list[int],
        tn_list: list[int],
        fn_list: list[int],
        latency_direct_list: list[float],
        latency_vectorq_list: list[float],
    ) -> Dict[str, Any]:
        # Mirrors the splitter benchmark statistics JSON structure.
        n = int(len(latency_vectorq_list))
        avg_latency_vcache_overall = float(sum(latency_vectorq_list) / n) if n > 0 else 0.0
        avg_latency_direct_overall = float(sum(latency_direct_list) / n) if n > 0 else 0.0

        hit_latencies_v = [
            latency_vectorq_list[i] for i in range(n) if int(cache_hit_list[i]) > 0
        ]
        miss_latencies_v = [
            latency_vectorq_list[i] for i in range(n) if int(cache_miss_list[i]) > 0
        ]
        hit_latencies_d = [
            latency_direct_list[i] for i in range(n) if int(cache_hit_list[i]) > 0
        ]
        miss_latencies_d = [
            latency_direct_list[i] for i in range(n) if int(cache_miss_list[i]) > 0
        ]

        avg_latency_vcache_cache_hit = (
            float(sum(hit_latencies_v) / len(hit_latencies_v)) if hit_latencies_v else 0.0
        )
        avg_latency_vcache_cache_miss = (
            float(sum(miss_latencies_v) / len(miss_latencies_v)) if miss_latencies_v else 0.0
        )
        avg_latency_direct_cache_hit = (
            float(sum(hit_latencies_d) / len(hit_latencies_d)) if hit_latencies_d else 0.0
        )
        avg_latency_direct_cache_miss = (
            float(sum(miss_latencies_d) / len(miss_latencies_d)) if miss_latencies_d else 0.0
        )

        cache_hit_rate_vcache = float(sum(cache_hit_list) / n) if n > 0 else 0.0
        cache_miss_rate_vcache = 1.0 - cache_hit_rate_vcache
        error_rate_vcache = float(sum(fp_list) / n) if n > 0 else 0.0

        duration_vcache = float(sum(latency_vectorq_list))
        duration_direct = float(sum(latency_direct_list))

        tp_sum = int(sum(tp_list))
        fp_sum = int(sum(fp_list))
        tn_sum = int(sum(tn_list))
        fn_sum = int(sum(fn_list))

        accuracy_vcache = float((tp_sum + tn_sum) / n) if n > 0 else 0.0
        precision_vcache = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
        recall_vcache = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
        f1_score_vcache = (
            float(2 * precision_vcache * recall_vcache / (precision_vcache + recall_vcache))
            if (precision_vcache + recall_vcache) > 0
            else 0.0
        )

        # Match splitter/benchmark4LM: "hits"/"misses" store the last step values, not totals.
        hits_last = int(cache_hit_list[-1]) if cache_hit_list else 0
        misses_last = int(cache_miss_list[-1]) if cache_miss_list else 0

        return {
            "avg_latency": {
                "cache": {
                    "overall": float(avg_latency_vcache_overall),
                    "cache_hit": float(avg_latency_vcache_cache_hit),
                    "cache_miss": float(avg_latency_vcache_cache_miss),
                },
                "direct": {"overall": float(avg_latency_direct_overall)},
                "difference": {
                    "overall": float(avg_latency_direct_overall - avg_latency_vcache_overall),
                    "cache_hit": float(avg_latency_direct_cache_hit - avg_latency_vcache_cache_hit),
                    "cache_miss": float(avg_latency_direct_cache_miss - avg_latency_vcache_cache_miss),
                },
                "ratio": {
                    "overall": float(avg_latency_direct_overall / avg_latency_vcache_overall)
                    if avg_latency_vcache_overall > 0
                    else "N/A",
                    "cache_hit": float(avg_latency_direct_cache_hit / avg_latency_vcache_cache_hit)
                    if avg_latency_vcache_cache_hit > 0
                    else "N/A",
                    "cache_miss": float(avg_latency_direct_cache_miss / avg_latency_vcache_cache_miss)
                    if avg_latency_vcache_cache_miss > 0
                    else "N/A",
                },
            },
            "cache": {
                "hit_rate": float(cache_hit_rate_vcache),
                "miss_rate": float(cache_miss_rate_vcache),
                "total_samples": int(n),
                "hits": int(hits_last),
                "misses": int(misses_last),
                "error_rate": float(error_rate_vcache),
            },
            "duration": {"vectorq": float(duration_vcache), "direct": float(duration_direct)},
            "statistics": {
                "accuracy": float(accuracy_vcache),
                "precision": float(precision_vcache),
                "recall": float(recall_vcache),
                "f1_score": float(f1_score_vcache),
            },
        }

    # Shared components across deltas
    # Use live BGE embeddings to match the splitter's base model
    shared_embedder = EmbeddingModel(device=args.device)
    embedding_engine = BGEEmbeddingEngine(embedding_model=shared_embedder)

    if args.similarity_evaluator == "string":
        similarity_evaluator = StringComparisonSimilarityEvaluator()
    else:
        similarity_evaluator = BenchmarkComparisonSimilarityEvaluator()

    aggregate_runs: List[Dict[str, Any]] = []

    for delta in deltas_to_run:
        # Build fresh vCache per delta (VerifiedDecisionPolicy has background threads / state)
        inference_engine = BenchmarkInferenceEngine()
        config = VCacheConfig(
            inference_engine=inference_engine,
            embedding_engine=embedding_engine,
            vector_db=HNSWLibVectorDB(
                similarity_metric_type=SimilarityMetricType.COSINE,
                max_capacity=args.max_capacity,
            ),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            eviction_policy=NoEvictionPolicy(),
            similarity_evaluator=similarity_evaluator,
        )

        timing: Optional[TimingCollector] = TimingCollector() if args.timing_output else None
        policy = VerifiedDecisionPolicy(delta=float(delta), timing_collector=timing)
        vcache = VCache(config=config, policy=policy)

        hits = 0
        tp = fp = tn = fn = 0
        n = 0
        per_sample_results: List[Dict[str, Any]] = []

        # Benchmark4LM-compatible lists
        cache_hit_list: list[int] = []
        cache_miss_list: list[int] = []
        tp_list: list[int] = []
        fp_list: list[int] = []
        tn_list: list[int] = []
        fn_list: list[int] = []
        latency_direct_list: list[float] = []
        latency_vectorq_list: list[float] = []

        hit_samples_f = None
        hit_samples_path = None
        if args.save_cache_hit_samples:
            hit_samples_path = os.path.abspath(str(args.save_cache_hit_samples))
            if len(deltas_to_run) > 1:
                hit_samples_path = _path_with_delta_suffix(hit_samples_path, delta=float(delta))
            hit_dir = os.path.dirname(hit_samples_path)
            if hit_dir:
                os.makedirs(hit_dir, exist_ok=True)
            hit_samples_f = open(hit_samples_path, "w", encoding="utf-8")

        t0 = time.time()
        desc_base = f"Evaluating delta={delta:g}"
        pbar = tqdm(rows, desc=desc_base, unit="samples")
        try:
            for r in pbar:
                prompt = r["prompt"]
                system_prompt = r.get("output_format", "") or ""
                id_set = _get_id_set(r)

                # Only require/consume response strings when using string-based correctness.
                if args.similarity_evaluator == "string":
                    label_response = r[args.llm_col]
                else:
                    # ID-based evaluation: we don't need response strings.
                    label_response = ""

                # Inject ground truth response for the benchmark engine.
                # NOTE: BenchmarkInferenceEngine expects `set_next_response()` to be called to set the attribute.
                inference_engine.set_next_response(label_response)

                step_t0 = time.time()
                is_hit, resp, resp_meta, nn_meta = vcache.infer_with_cache_info(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    id_set=id_set,
                )
                step_latency = float(time.time() - step_t0)

                n += 1
                hits += int(is_hit)
                try:
                    pbar.set_description(f"{desc_base} hits={hits}/{n} ({(hits/max(1,n)):.1%})")
                except Exception:
                    pass

                d_tp, d_fp, d_tn, d_fn = _score_step(
                    is_cache_hit=is_hit,
                    label_id_set=id_set,
                    label_response=label_response,
                    cache_response=resp,
                    response_metadata=resp_meta,
                    nn_metadata=nn_meta,
                )
                tp += d_tp
                fp += d_fp
                tn += d_tn
                fn += d_fn

                cache_hit_list.append(int(bool(is_hit)))
                cache_miss_list.append(int(not bool(is_hit)))
                tp_list.append(int(d_tp))
                fp_list.append(int(d_fp))
                tn_list.append(int(d_tn))
                fn_list.append(int(d_fn))
                # No true direct baseline here; keep parity with splitter benchmark (same latency list)
                latency_direct_list.append(float(step_latency))
                latency_vectorq_list.append(float(step_latency))

                # Optional: dump every cache hit sample to JSONL for later inspection.
                if bool(is_hit) and hit_samples_f is not None:
                    rec = {
                        "sample_index": int(n),
                        "delta": float(delta),
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "label_id_set": int(id_set),
                        "label_response": label_response,
                        "cached_embedding_id": int(getattr(resp_meta, "embedding_id", -1)),
                        "cached_id_set": int(getattr(resp_meta, "id_set", -1)),
                        "cached_prompt": getattr(resp_meta, "prompt", "") or "",
                        "cached_response": resp,
                        "t_hat": getattr(resp_meta, "t_hat", None),
                        "t_prime": getattr(resp_meta, "t_prime", None),
                        "gamma": getattr(resp_meta, "gamma", None),
                        "var_t": getattr(resp_meta, "var_t", None),
                    }
                    hit_samples_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                per_sample_results.append(
                    {
                        "sample_index": n,
                        "is_hit": bool(is_hit),
                        "running_hit_rate": hits / max(1, n),
                        "tp": d_tp,
                        "fp": d_fp,
                        "tn": d_tn,
                        "fn": d_fn,
                    }
                )

                # VerifiedDecisionPolicy updates cache metadata asynchronously; a tiny sleep (as in benchmark.py)
                # prevents the evaluation loop from outrunning the background worker and biasing toward EXPLORE.
                if args.sleep and args.sleep > 0:
                    time.sleep(float(args.sleep))
        finally:
            if hit_samples_f is not None:
                try:
                    hit_samples_f.close()
                    print(f"Cache-hit samples saved to {hit_samples_path}")
                except Exception:
                    pass

        elapsed = time.time() - t0

        print(f"dataset={args.dataset}")
        print(f"columns: embedding={args.embedding_col} llm={args.llm_col}")
        print(f"delta={float(delta)} n={n} time={elapsed:.2f}s")
        print(f"hit_rate={hits}/{n} ({(hits/max(1,n)):.1%})")
        print(f"tp={tp} fp={fp} tn={tn} fn={fn}")

        run_output_path = None
        if args.output_json:
            run_output_path = os.path.abspath(str(args.output_json))
            if len(deltas_to_run) > 1:
                run_output_path = _path_with_delta_suffix(run_output_path, delta=float(delta))
            out_dir = os.path.dirname(run_output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            output_data = {
                "args": {**vars(args), "delta": float(delta)},
                "summary": {
                    "delta": float(delta),
                    "n": n,
                    "hits": hits,
                    "hit_rate": hits / max(1, n),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "total_time": elapsed,
                },
                "per_sample": per_sample_results,
            }
            with open(run_output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {run_output_path}")

        # Timing summary (embedding/retrieval/total) for this delta.
        if args.timing_output:
            timing_path = os.path.abspath(str(args.timing_output))
            timing_dir = os.path.dirname(timing_path)
            if timing_dir:
                os.makedirs(timing_dir, exist_ok=True)
            embedding_s = float(timing.get("embedding") if timing is not None else 0.0)
            retrieval_s = float(timing.get("retrieval") if timing is not None else 0.0)
            total_s = float(sum(latency_vectorq_list))
            with open(timing_path, "a", encoding="utf-8") as f:
                f.write(
                    "dataset={dataset} delta={delta} n={n} embedding_s={emb:.6f} retrieval_s={ret:.6f} total_s={total:.6f}\n".format(
                        dataset=str(args.dataset),
                        delta=float(delta),
                        n=int(n),
                        emb=embedding_s,
                        ret=retrieval_s,
                        total=total_s,
                    )
                )

        # Default: always write benchmark4LM-compatible output (results_<ts>.json + statistics_<ts>.json)
        bench_dir = _resolve_benchmark_output_dir(delta=float(delta))
        observations_dict: Dict[str, Dict[str, float]] = {}
        gammas_dict: Dict[str, float] = {}
        t_hats_dict: Dict[str, float] = {}
        t_primes_dict: Dict[str, float] = {}
        var_ts_dict: Dict[str, float] = {}

        try:
            metadata_objects = (
                vcache.vcache_config.embedding_metadata_storage.get_all_embedding_metadata_objects()
            )
        except Exception:
            metadata_objects = []

        for metadata_object in metadata_objects:
            try:
                embedding_id = str(getattr(metadata_object, "embedding_id"))
            except Exception:
                continue
            observations_dict[embedding_id] = getattr(metadata_object, "observations", {})
            gammas_dict[embedding_id] = getattr(metadata_object, "gamma", None)
            t_hats_dict[embedding_id] = getattr(metadata_object, "t_hat", None)
            t_primes_dict[embedding_id] = getattr(metadata_object, "t_prime", None)
            var_ts_dict[embedding_id] = getattr(metadata_object, "var_t", None)

        try:
            global_observations_dict = vcache.vcache_policy.global_observations
            global_gamma = vcache.vcache_policy.bayesian.global_gamma
            global_t_hat = vcache.vcache_policy.bayesian.global_t_hat
            global_t_prime = vcache.vcache_policy.bayesian.global_t_prime
            global_var_t = vcache.vcache_policy.bayesian.global_var_t
        except Exception:
            global_observations_dict = {}
            global_gamma = None
            global_t_hat = None
            global_t_prime = None
            global_var_t = None

        bench_data: Dict[str, Any] = {
            "config": {
                "filepath": str(args.dataset),
                "embedding_model": str(args.embedding_col or ""),
                "llm_model": str(args.llm_col or ""),
                "eviction_policy": str(NoEvictionPolicy()),
                "is_static_threshold": False,
                "threshold": None,
                "delta": float(delta),
            },
            "cache_hit_list": cache_hit_list,
            "cache_miss_list": cache_miss_list,
            "tp_list": tp_list,
            "fp_list": fp_list,
            "tn_list": tn_list,
            "fn_list": fn_list,
            "latency_direct_list": latency_direct_list,
            "latency_vectorq_list": latency_vectorq_list,
            "observations_dict": observations_dict,
            "gammas_dict": gammas_dict,
            "t_hats_dict": t_hats_dict,
            "t_primes_dict": t_primes_dict,
            "var_ts_dict": var_ts_dict,
            "global_observations_dict": global_observations_dict,
            "global_gamma": global_gamma,
            "global_t_hat": global_t_hat,
            "global_t_prime": global_t_prime,
            "global_var_t": global_var_t,
        }

        bench_path = os.path.join(bench_dir, f"results_{benchmark_timestamp}.json")
        with open(bench_path, "w") as f:
            json.dump(bench_data, f, indent=4)
        print(f"Benchmark-format results saved to {bench_path}")

        statistics_path = os.path.join(bench_dir, f"statistics_{benchmark_timestamp}.json")
        statistics_data = _compute_statistics_json(
            cache_hit_list=cache_hit_list,
            cache_miss_list=cache_miss_list,
            tp_list=tp_list,
            fp_list=fp_list,
            tn_list=tn_list,
            fn_list=fn_list,
            latency_direct_list=latency_direct_list,
            latency_vectorq_list=latency_vectorq_list,
        )
        with open(statistics_path, "w") as f:
            json.dump(statistics_data, f, indent=4)
        print(f"Benchmark-format statistics saved to {statistics_path}")

        aggregate_runs.append(
            {
                "delta": float(delta),
                "n": n,
                "hits": hits,
                "hit_rate": hits / max(1, n),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "total_time": elapsed,
                "output_json": run_output_path,
                "cache_hit_samples": hit_samples_path,
            }
        )

        # Clean shutdown (VerifiedDecisionPolicy uses background threads)
        time.sleep(0.1)
        vcache.vcache_policy.shutdown()

    # If user requested multiple deltas and provided --output-json, also write an aggregate summary file.
    if args.aggregate_json and len(deltas_to_run) > 1:
        agg_path = os.path.abspath(str(args.aggregate_json))
        out_dir = os.path.dirname(agg_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump(
                {"args": vars(args), "runs": aggregate_runs},
                f,
                indent=2,
            )
        print(f"Aggregate results saved to {agg_path}")


if __name__ == "__main__":
    main()


