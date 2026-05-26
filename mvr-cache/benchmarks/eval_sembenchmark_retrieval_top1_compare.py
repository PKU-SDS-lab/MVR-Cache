"""
Compare retrieval top-1 quality for:
1. Single-vector HNSW over one pooled embedding per prompt.
2. Multi-vector HNSW over splitter segment vectors, followed by MaxSim reranking.

This benchmark is retrieval-only. It does not run the verified Bayesian hit/miss policy.
Instead, it simulates an insert-all stream:
  - for each sample, query both indexes against previously inserted prompts
  - evaluate whether the returned top-1 matches the ground-truth group (`id_set`)
  - then insert the current sample into both indexes

The multi-vector path mirrors the splitter code path:
  - encode the query once with `MaxSimSplitter.encode_text`
  - build the query multivector via `split_text_return_maxsim_tensor_from_encoded`
  - query the multivector HNSW with every query vector
  - union candidate parent ids
  - rerank only those candidates with MaxSim
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import DownloadConfig, load_dataset
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Ensure we import the local repo copy, not any installed/sibling package.
# --------------------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _purge_nonlocal_vcache_modules() -> None:
    local_prefix = os.path.abspath(_PROJECT_ROOT) + os.sep
    for module_name, module in list(sys.modules.items()):
        if module_name != "vcache" and not module_name.startswith("vcache."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and not os.path.abspath(module_file).startswith(local_prefix):
            del sys.modules[module_name]


_purge_nonlocal_vcache_modules()

from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
    HNSWLibVectorDB,
)
from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
)
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter
from vcache.vcache_policy.strategies.verified_splitter import (
    VerifiedSplitterDecisionPolicy,
    _MultiVectorHNSWIndex,
)

# Guard against accidentally importing a non-local splitter implementation.
if "max_segments" not in MaxSimSplitter.__init__.__code__.co_varnames:
    _purge_nonlocal_vcache_modules()
    MaxSimSplitter = importlib.import_module(
        "vcache.vcache_core.splitter.MaxSimSplitter"
    ).MaxSimSplitter


@dataclass
class IndexedDoc:
    embedding_id: int
    prompt: str
    id_set: int
    pooled_knn: Any
    pooled_no_cls: Any
    maxsim_tensor: Any


class TimingCollector:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}

    def add(self, name: str, dt_seconds: float) -> None:
        k = str(name)
        self.sums[k] = self.sums.get(k, 0.0) + float(dt_seconds)

    def get(self, name: str) -> float:
        return float(self.sums.get(str(name), 0.0))


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
    if v == -1:
        v = row.get("label_id", -1)
    try:
        return int(v)
    except Exception:
        return -1


def _load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
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
        if "prompt" not in df.columns:
            raise ValueError(
                f"Local dataset is missing required column 'prompt'. Available columns: {list(df.columns)}"
            )
        rows = df.to_dict("records")
    else:
        cache_paths = _ensure_hf_cache_env(args.hf_cache_base)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACEHUB_API_TOKEN"
        )
        split = "train"
        if args.max_samples is not None:
            split = f"train[:{args.max_samples}]"
        dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
        rows = load_dataset(
            args.dataset,
            split=split,
            cache_dir=cache_paths["DATASETS_CACHE"],
            token=hf_token,
            download_config=dl_cfg,
        )
    if args.max_samples is not None:
        rows = list(rows)[: int(args.max_samples)]
    else:
        rows = list(rows)
    return rows


def _score_multivector_candidate(
    *,
    query_enc: dict,
    query_tensor,
    candidate: IndexedDoc,
    mix_fullcos: bool,
) -> float:
    maxsim01 = VerifiedSplitterDecisionPolicy._maxsim_from_tensors(
        query_tensor, candidate.maxsim_tensor
    )
    if not mix_fullcos:
        return float(maxsim01)
    fullcos01 = VerifiedSplitterDecisionPolicy._cos01(
        query_enc["pooled_no_cls"], candidate.pooled_no_cls
    )
    return 0.5 * (float(maxsim01) + float(fullcos01))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset id or local .csv/.parquet file path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate. If omitted, use the full dataset.",
    )
    parser.add_argument(
        "--splitter-checkpoint",
        required=True,
        help="MaxSimSplitter checkpoint file or directory.",
    )
    parser.add_argument(
        "--splitter-device",
        default="cpu",
        help="Device for MaxSimSplitter / embedding model (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--splitter-max-segments",
        type=int,
        default=4,
        help="Max segments used by the RL splitter.",
    )
    parser.add_argument(
        "--splitter-overlap-tokens",
        type=int,
        default=0,
        help="Token overlap used when reconstructing segment embeddings.",
    )
    parser.add_argument(
        "--include-full-embedding",
        action="store_true",
        help="Append the full pooled embedding as an extra MaxSim vector row.",
    )
    parser.add_argument(
        "--mix-fullcos",
        action="store_true",
        help="Use 0.5*(MaxSim + cosine(full_embed_no_cls)) during multivector reranking.",
    )
    parser.add_argument(
        "--multivector-top-k",
        type=int,
        default=10,
        help="Top-k retrieved per query vector from the multivector HNSW.",
    )
    parser.add_argument(
        "--single-max-capacity",
        type=int,
        default=200_000,
        help="Capacity of the single-vector HNSW in documents.",
    )
    parser.add_argument(
        "--multivector-max-elements",
        type=int,
        default=2_000_000,
        help="Capacity of the multivector HNSW in vectors.",
    )
    parser.add_argument(
        "--hf-cache-base",
        default=os.environ.get("HF_CACHE_BASE", None),
        help="Base dir for HuggingFace caches.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write full results JSON to this path.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    rows = _load_rows(args)

    shared_embedder = EmbeddingModel(device=args.splitter_device)
    splitter = MaxSimSplitter(
        checkpoint_path=args.splitter_checkpoint,
        device=args.splitter_device,
        embedding_model=shared_embedder,
        max_segments=int(args.splitter_max_segments),
        overlap_tokens=int(args.splitter_overlap_tokens),
        include_full_embedding=bool(args.include_full_embedding),
    )

    single_index = HNSWLibVectorDB(
        similarity_metric_type=SimilarityMetricType.COSINE,
        max_capacity=int(args.single_max_capacity),
    )
    multivector_index = _MultiVectorHNSWIndex(
        max_elements=int(args.multivector_max_elements)
    )

    docs_by_id: Dict[int, IndexedDoc] = {}
    seen_id_set_counts: Dict[int, int] = {}
    timing = TimingCollector()
    per_sample: List[Dict[str, Any]] = []

    single_correct_all = 0
    multivector_correct_all = 0
    multivector_candidate_hit_all = 0
    eligible_samples = 0
    single_correct_eligible = 0
    multivector_correct_eligible = 0
    multivector_candidate_hit_eligible = 0
    both_correct = 0
    single_only_correct = 0
    multivector_only_correct = 0
    both_wrong = 0
    total_candidate_count = 0

    pbar = tqdm(rows, desc="Comparing retrieval top-1", unit="samples")
    for sample_index, row in enumerate(pbar, start=1):
        prompt = row["prompt"]
        id_set = _get_id_set(row)
        eligible = id_set != -1 and seen_id_set_counts.get(id_set, 0) > 0
        if eligible:
            eligible_samples += 1

        t0 = time.time()
        query_enc = splitter.encode_text(prompt)
        timing.add("encode_s", time.time() - t0)

        pooled_knn_cpu = query_enc["pooled_knn"].detach().float().cpu().tolist()

        t0 = time.time()
        single_knn = single_index.get_knn(pooled_knn_cpu, k=1)
        timing.add("single_retrieval_s", time.time() - t0)

        single_top1_score = None
        single_top1_embedding_id = None
        single_top1_id_set = None
        single_top1_prompt = None
        single_top1_correct = False
        if single_knn:
            single_top1_score, single_top1_embedding_id = single_knn[0]
            single_doc = docs_by_id.get(int(single_top1_embedding_id))
            if single_doc is not None:
                single_top1_id_set = int(single_doc.id_set)
                single_top1_prompt = single_doc.prompt
                single_top1_correct = bool(id_set != -1 and single_doc.id_set == id_set)

        if single_top1_correct:
            single_correct_all += 1
            if eligible:
                single_correct_eligible += 1

        t0 = time.time()
        query_tensor = splitter.split_text_return_maxsim_tensor_from_encoded(query_enc)
        query_vectors_np = query_tensor.detach().float().cpu().numpy()
        timing.add("multi_query_tensor_s", time.time() - t0)

        t0 = time.time()
        multivector_parent_ids = multivector_index.query_candidate_parents(
            query_vectors=query_vectors_np,
            k_per_vector=max(1, int(args.multivector_top_k)),
        )
        timing.add("multi_retrieval_s", time.time() - t0)

        multivector_candidates: List[IndexedDoc] = []
        for parent_id in multivector_parent_ids:
            doc = docs_by_id.get(int(parent_id))
            if doc is not None:
                multivector_candidates.append(doc)

        multivector_candidate_count = len(multivector_candidates)
        total_candidate_count += multivector_candidate_count
        multivector_candidate_hit = bool(
            id_set != -1 and any(doc.id_set == id_set for doc in multivector_candidates)
        )
        if multivector_candidate_hit:
            multivector_candidate_hit_all += 1
            if eligible:
                multivector_candidate_hit_eligible += 1

        multivector_top1_score = None
        multivector_top1_embedding_id = None
        multivector_top1_id_set = None
        multivector_top1_prompt = None
        multivector_top1_correct = False

        t0 = time.time()
        best_doc: Optional[IndexedDoc] = None
        best_score = -1.0
        for candidate in multivector_candidates:
            score = _score_multivector_candidate(
                query_enc=query_enc,
                query_tensor=query_tensor,
                candidate=candidate,
                mix_fullcos=bool(args.mix_fullcos),
            )
            if score > best_score:
                best_score = score
                best_doc = candidate
        timing.add("multi_rerank_s", time.time() - t0)

        if best_doc is not None:
            multivector_top1_score = float(best_score)
            multivector_top1_embedding_id = int(best_doc.embedding_id)
            multivector_top1_id_set = int(best_doc.id_set)
            multivector_top1_prompt = best_doc.prompt
            multivector_top1_correct = bool(id_set != -1 and best_doc.id_set == id_set)

        if multivector_top1_correct:
            multivector_correct_all += 1
            if eligible:
                multivector_correct_eligible += 1

        if eligible:
            if single_top1_correct and multivector_top1_correct:
                both_correct += 1
            elif single_top1_correct and not multivector_top1_correct:
                single_only_correct += 1
            elif multivector_top1_correct and not single_top1_correct:
                multivector_only_correct += 1
            else:
                both_wrong += 1

        per_sample.append(
            {
                "sample_index": int(sample_index),
                "id_set": int(id_set),
                "eligible": bool(eligible),
                "single_top1_correct": bool(single_top1_correct),
                "single_top1_score": (
                    None if single_top1_score is None else float(single_top1_score)
                ),
                "single_top1_embedding_id": single_top1_embedding_id,
                "single_top1_id_set": single_top1_id_set,
                "single_top1_prompt": single_top1_prompt,
                "multivector_candidate_count": int(multivector_candidate_count),
                "multivector_candidate_hit": bool(multivector_candidate_hit),
                "multivector_top1_correct": bool(multivector_top1_correct),
                "multivector_top1_score": (
                    None
                    if multivector_top1_score is None
                    else float(multivector_top1_score)
                ),
                "multivector_top1_embedding_id": multivector_top1_embedding_id,
                "multivector_top1_id_set": multivector_top1_id_set,
                "multivector_top1_prompt": multivector_top1_prompt,
                "prompt": prompt,
            }
        )

        embedding_id = single_index.add(pooled_knn_cpu)
        doc = IndexedDoc(
            embedding_id=int(embedding_id),
            prompt=prompt,
            id_set=int(id_set),
            pooled_knn=query_enc["pooled_knn"].detach(),
            pooled_no_cls=query_enc["pooled_no_cls"].detach(),
            maxsim_tensor=query_tensor.detach(),
        )
        docs_by_id[int(embedding_id)] = doc
        multivector_index.add_parent_vectors(
            parent_id=int(embedding_id),
            vectors=query_vectors_np,
        )
        if id_set != -1:
            seen_id_set_counts[id_set] = seen_id_set_counts.get(id_set, 0) + 1

        pbar.set_description(
            "Comparing retrieval top-1 "
            f"single={single_correct_all}/{sample_index} "
            f"multi={multivector_correct_all}/{sample_index}"
        )

    total_samples = len(per_sample)
    avg_candidate_count = (
        float(total_candidate_count / total_samples) if total_samples > 0 else 0.0
    )
    single_top1_accuracy_all = (
        float(single_correct_all / total_samples) if total_samples > 0 else 0.0
    )
    multivector_top1_accuracy_all = (
        float(multivector_correct_all / total_samples) if total_samples > 0 else 0.0
    )
    multivector_candidate_hit_rate_all = (
        float(multivector_candidate_hit_all / total_samples) if total_samples > 0 else 0.0
    )
    single_top1_accuracy_eligible = (
        float(single_correct_eligible / eligible_samples) if eligible_samples > 0 else 0.0
    )
    multivector_top1_accuracy_eligible = (
        float(multivector_correct_eligible / eligible_samples)
        if eligible_samples > 0
        else 0.0
    )
    multivector_candidate_hit_rate_eligible = (
        float(multivector_candidate_hit_eligible / eligible_samples)
        if eligible_samples > 0
        else 0.0
    )

    summary = {
        "total_samples": int(total_samples),
        "eligible_samples": int(eligible_samples),
        "single_top1_accuracy_all": float(single_top1_accuracy_all),
        "single_top1_accuracy_eligible": float(single_top1_accuracy_eligible),
        "multivector_top1_accuracy_all": float(multivector_top1_accuracy_all),
        "multivector_top1_accuracy_eligible": float(
            multivector_top1_accuracy_eligible
        ),
        "multivector_candidate_hit_rate_all": float(multivector_candidate_hit_rate_all),
        "multivector_candidate_hit_rate_eligible": float(
            multivector_candidate_hit_rate_eligible
        ),
        "pairwise_eligible": {
            "both_correct": int(both_correct),
            "single_only_correct": int(single_only_correct),
            "multivector_only_correct": int(multivector_only_correct),
            "both_wrong": int(both_wrong),
        },
        "avg_multivector_candidate_count": float(avg_candidate_count),
        "timing": {
            "encode_s": float(timing.get("encode_s")),
            "single_retrieval_s": float(timing.get("single_retrieval_s")),
            "multi_query_tensor_s": float(timing.get("multi_query_tensor_s")),
            "multi_retrieval_s": float(timing.get("multi_retrieval_s")),
            "multi_rerank_s": float(timing.get("multi_rerank_s")),
        },
    }

    results = {
        "args": {
            "dataset": str(args.dataset),
            "max_samples": args.max_samples,
            "splitter_checkpoint": str(args.splitter_checkpoint),
            "splitter_device": str(args.splitter_device),
            "splitter_max_segments": int(args.splitter_max_segments),
            "splitter_overlap_tokens": int(args.splitter_overlap_tokens),
            "include_full_embedding": bool(args.include_full_embedding),
            "mix_fullcos": bool(args.mix_fullcos),
            "multivector_top_k": int(args.multivector_top_k),
            "single_max_capacity": int(args.single_max_capacity),
            "multivector_max_elements": int(args.multivector_max_elements),
        },
        "summary": summary,
        "per_sample": per_sample,
    }

    print(f"dataset={args.dataset}")
    print(f"total_samples={total_samples}")
    print(f"eligible_samples={eligible_samples}")
    print(
        "single_top1_accuracy_all={:.4f} single_top1_accuracy_eligible={:.4f}".format(
            single_top1_accuracy_all, single_top1_accuracy_eligible
        )
    )
    print(
        "multivector_top1_accuracy_all={:.4f} multivector_top1_accuracy_eligible={:.4f}".format(
            multivector_top1_accuracy_all, multivector_top1_accuracy_eligible
        )
    )
    print(
        "pairwise_eligible both_correct={both_correct} single_only={single_only} "
        "multivector_only={multi_only} both_wrong={both_wrong}".format(
            both_correct=both_correct,
            single_only=single_only_correct,
            multi_only=multivector_only_correct,
            both_wrong=both_wrong,
        )
    )

    if args.output_json:
        out_path = os.path.abspath(str(args.output_json))
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
