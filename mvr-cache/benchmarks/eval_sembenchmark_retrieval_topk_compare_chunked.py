"""
Compare retrieval quality for:
1. Single-vector full scan over one pooled embedding per prompt.
2. Multi-vector HNSW over splitter segment vectors, followed by MaxSim reranking.

This benchmark is retrieval-only. It does not run the verified Bayesian hit/miss policy.
Instead, it simulates an insert-all stream:
  - for each sample, query both indexes against previously inserted prompts
  - evaluate top-1 correctness and top-k candidate quality against the ground-truth group (`id_set`)
  - then insert the current sample into both indexes

Multi-vector path details:
  - single-vector baseline still uses one pooled embedding from `MaxSimSplitter.encode_text`
  - multivector path uses chunked prompt processing for long prompts
  - each chunk is segmented with the splitter, and all chunk tensors are concatenated
  - query each multivector row against multivector HNSW
  - union candidate parent ids
  - rerank candidates with the same MaxSim implementation used in the splitter policy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import importlib
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import DownloadConfig, load_dataset
from tqdm import tqdm

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

from vcache.vcache_core.splitter.embedding_model import EmbeddingModel  # noqa: E402
from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter  # noqa: E402
from vcache.vcache_policy.strategies.verified_splitter import (  # noqa: E402
    VerifiedSplitterDecisionPolicy,
    _MultiVectorHNSWIndex,
)

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
    gt_key: Optional[str]
    pooled_knn: np.ndarray
    pooled_no_cls: Any
    maxsim_tensor: Any


class TimingCollector:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}

    def add(self, name: str, dt_seconds: float) -> None:
        key = str(name)
        self.sums[key] = self.sums.get(key, 0.0) + float(dt_seconds)

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
    value = row.get("id_set", -1)
    if value == -1:
        value = row.get("ID_Set", -1)
    if value == -1:
        value = row.get("label_id", -1)
    try:
        return int(value)
    except Exception:
        return -1


def _normalize_answer_text(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .rstrip(".")
        .lower()
        .replace('"', "")
        .replace("'", "")
        .replace("[", "")
        .replace("]", "")
    )


def _resolve_ground_truth_mode(
    rows: List[Dict[str, Any]], args: argparse.Namespace
) -> str:
    requested = str(args.ground_truth_mode).strip().lower()
    if requested not in {"auto", "id_set", "string"}:
        raise ValueError(
            f"Unsupported --ground-truth-mode={args.ground_truth_mode!r}. "
            "Expected one of: auto, id_set, string."
        )

    probe_rows = rows[: min(len(rows), 256)]
    has_id_labels = any(_get_id_set(row) != -1 for row in probe_rows)

    if requested == "auto":
        if has_id_labels:
            return "id_set"
        if args.llm_col:
            return "string"
        raise ValueError(
            "Could not infer ground-truth mode: dataset has no usable id_set/ID_Set/label_id "
            "and --llm-col was not provided for string matching."
        )

    if requested == "id_set" and not has_id_labels:
        raise ValueError(
            "Requested --ground-truth-mode id_set, but dataset has no usable id_set/ID_Set/label_id values."
        )
    if requested == "string" and not args.llm_col:
        raise ValueError(
            "Requested --ground-truth-mode string, but --llm-col was not provided."
        )
    return requested


def _get_ground_truth_key(
    row: Dict[str, Any], *, ground_truth_mode: str, llm_col: Optional[str]
) -> Optional[str]:
    if ground_truth_mode == "id_set":
        id_set = _get_id_set(row)
        return None if id_set == -1 else f"id::{id_set}"
    if ground_truth_mode == "string":
        norm = _normalize_answer_text(row.get(str(llm_col or ""), None))
        return None if not norm else f"str::{norm}"
    raise ValueError(f"Unsupported ground_truth_mode={ground_truth_mode!r}")


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


def _build_chunked_maxsim_tensor(
    *,
    splitter: MaxSimSplitter,
    text: str,
    chunk_token_limit: int,
    timing: TimingCollector | None = None,
    timing_prefix: str = "",
) -> Any:
    chunk_texts = _chunk_text_by_tokens(
        text=text,
        tokenizer=splitter.generator.tokenizer,
        chunk_token_limit=chunk_token_limit,
    )
    if not chunk_texts:
        chunk_texts = [text]

    chunk_tensors = []
    for chunk_text in chunk_texts:
        t0 = time.time()
        chunk_enc = splitter.encode_text(chunk_text)
        if timing is not None:
            timing.add(f"{timing_prefix}chunk_encode_s", time.time() - t0)

        t0 = time.time()
        chunk_tensor = splitter.split_text_return_maxsim_tensor_from_encoded(chunk_enc)
        if timing is not None:
            timing.add(f"{timing_prefix}chunk_segment_s", time.time() - t0)
        chunk_tensors.append(chunk_tensor.detach())

    if len(chunk_tensors) == 1:
        return chunk_tensors[0]

    import torch as _torch

    return _torch.cat(chunk_tensors, dim=0)


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


def _count_correct_docs(docs: List[IndexedDoc], gt_key: Optional[str]) -> int:
    if not gt_key:
        return 0
    return int(sum(1 for doc in docs if doc.gt_key == gt_key))


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _single_full_scan_topk(
    *,
    query_vector: np.ndarray,
    docs_by_id: Dict[int, IndexedDoc],
    k: int,
) -> List[tuple[float, IndexedDoc]]:
    if not docs_by_id:
        return []

    scored = [
        (_cosine_similarity(query_vector, doc.pooled_knn), doc)
        for doc in docs_by_id.values()
    ]
    return heapq.nlargest(max(1, int(k)), scored, key=lambda item: item[0])


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
        "--eval-top-k",
        type=int,
        default=10,
        help="Number of closest top-k candidates to evaluate for both retrieval paths.",
    )
    parser.add_argument(
        "--chunk-token-limit",
        type=int,
        default=448,
        help="Chunk size used to build multivector representations for long prompts.",
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
        "--ground-truth-mode",
        default="auto",
        help="How to decide correctness: auto, id_set, or string.",
    )
    parser.add_argument(
        "--llm-col",
        default=None,
        help="Response column to use when --ground-truth-mode string is selected.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write full results JSON to this path.",
    )
    args = parser.parse_args()
    eval_top_k = max(1, int(args.eval_top_k))
    report_top20_k = 20
    retrieval_depth = max(eval_top_k, report_top20_k)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    rows = _load_rows(args)
    ground_truth_mode = _resolve_ground_truth_mode(rows, args)
    if ground_truth_mode == "string":
        available_columns = list(rows[0].keys()) if rows else []
        if not args.llm_col or args.llm_col not in available_columns:
            raise ValueError(
                f"String ground truth requires --llm-col to be present in the dataset. "
                f"Requested={args.llm_col!r}, available columns={available_columns}"
            )

    shared_embedder = EmbeddingModel(device=args.splitter_device)
    splitter = MaxSimSplitter(
        checkpoint_path=args.splitter_checkpoint,
        device=args.splitter_device,
        embedding_model=shared_embedder,
        max_segments=int(args.splitter_max_segments),
        overlap_tokens=int(args.splitter_overlap_tokens),
        include_full_embedding=bool(args.include_full_embedding),
    )

    multivector_index = _MultiVectorHNSWIndex(
        max_elements=int(args.multivector_max_elements)
    )

    docs_by_id: Dict[int, IndexedDoc] = {}
    next_embedding_id = 0
    seen_gt_key_counts: Dict[str, int] = {}
    timing = TimingCollector()
    per_sample: List[Dict[str, Any]] = []

    single_top1_correct_all = 0
    multivector_top1_correct_all = 0
    single_topk_hit_all = 0
    multivector_topk_hit_all = 0
    single_topk_correct_total_all = 0
    multivector_topk_correct_total_all = 0
    eligible_samples = 0
    single_top1_correct_eligible = 0
    multivector_top1_correct_eligible = 0
    single_top20_hit_all = 0
    multivector_top20_hit_all = 0
    single_top20_hit_eligible = 0
    multivector_top20_hit_eligible = 0
    single_topk_hit_eligible = 0
    multivector_topk_hit_eligible = 0
    single_topk_correct_total_eligible = 0
    multivector_topk_correct_total_eligible = 0
    both_top1_correct = 0
    single_only_top1_correct = 0
    multivector_only_top1_correct = 0
    both_top1_wrong = 0
    both_topk_hit = 0
    single_only_topk_hit = 0
    multivector_only_topk_hit = 0
    both_topk_miss = 0
    total_multivector_candidate_count = 0
    total_multivector_rerank_count = 0

    pbar = tqdm(rows, desc="Comparing retrieval top-k", unit="samples")
    for sample_index, row in enumerate(pbar, start=1):
        prompt = row["prompt"]
        id_set = _get_id_set(row)
        gt_key = _get_ground_truth_key(
            row,
            ground_truth_mode=ground_truth_mode,
            llm_col=args.llm_col,
        )
        eligible = bool(gt_key) and seen_gt_key_counts.get(gt_key, 0) > 0
        if eligible:
            eligible_samples += 1

        t0 = time.time()
        query_enc = splitter.encode_text(prompt)
        timing.add("encode_s", time.time() - t0)

        pooled_knn_np = query_enc["pooled_knn"].detach().float().cpu().numpy()

        t0 = time.time()
        single_topk_scored = _single_full_scan_topk(
            query_vector=pooled_knn_np,
            docs_by_id=docs_by_id,
            k=int(retrieval_depth),
        )
        timing.add("single_full_scan_s", time.time() - t0)

        single_topk_docs: List[IndexedDoc] = []
        for _score, doc in single_topk_scored[:eval_top_k]:
            single_topk_docs.append(doc)
        single_top20_docs = [doc for _score, doc in single_topk_scored[:report_top20_k]]

        single_top1_score = None
        single_top1_embedding_id = None
        single_top1_id_set = None
        single_top1_prompt = None
        single_top1_correct = False
        if single_topk_scored:
            single_top1_score, single_doc = single_topk_scored[0]
            single_top1_embedding_id = int(single_doc.embedding_id)
            single_top1_id_set = int(single_doc.id_set)
            single_top1_prompt = single_doc.prompt
            single_top1_correct = bool(gt_key and single_doc.gt_key == gt_key)

        single_topk_correct_count = _count_correct_docs(single_topk_docs, gt_key)
        single_topk_hit = bool(single_topk_correct_count > 0)
        single_top20_correct_count = _count_correct_docs(single_top20_docs, gt_key)
        single_top20_hit = bool(single_top20_correct_count > 0)

        if single_top1_correct:
            single_top1_correct_all += 1
            if eligible:
                single_top1_correct_eligible += 1
        if single_topk_hit:
            single_topk_hit_all += 1
            if eligible:
                single_topk_hit_eligible += 1
        if single_top20_hit:
            single_top20_hit_all += 1
            if eligible:
                single_top20_hit_eligible += 1
        single_topk_correct_total_all += int(single_topk_correct_count)
        if eligible:
            single_topk_correct_total_eligible += int(single_topk_correct_count)

        t0 = time.time()
        query_tensor = _build_chunked_maxsim_tensor(
            splitter=splitter,
            text=prompt,
            chunk_token_limit=int(args.chunk_token_limit),
            timing=timing,
            timing_prefix="query_",
        )
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
        total_multivector_candidate_count += multivector_candidate_count

        t0 = time.time()
        scored_candidates: List[tuple[float, IndexedDoc]] = []
        for candidate in multivector_candidates:
            score = _score_multivector_candidate(
                query_enc=query_enc,
                query_tensor=query_tensor,
                candidate=candidate,
                mix_fullcos=bool(args.mix_fullcos),
            )
            scored_candidates.append((float(score), candidate))
        timing.add("multi_rerank_s", time.time() - t0)

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        multivector_topk_scored = scored_candidates[:eval_top_k]
        multivector_topk_docs = [doc for _, doc in multivector_topk_scored]
        multivector_top20_docs = [doc for _, doc in scored_candidates[:report_top20_k]]
        total_multivector_rerank_count += len(multivector_topk_docs)

        multivector_top1_score = None
        multivector_top1_embedding_id = None
        multivector_top1_id_set = None
        multivector_top1_prompt = None
        multivector_top1_correct = False
        if multivector_topk_scored:
            multivector_top1_score, best_doc = multivector_topk_scored[0]
            multivector_top1_embedding_id = int(best_doc.embedding_id)
            multivector_top1_id_set = int(best_doc.id_set)
            multivector_top1_prompt = best_doc.prompt
            multivector_top1_correct = bool(gt_key and best_doc.gt_key == gt_key)

        multivector_topk_correct_count = _count_correct_docs(multivector_topk_docs, gt_key)
        multivector_topk_hit = bool(multivector_topk_correct_count > 0)
        multivector_top20_correct_count = _count_correct_docs(
            multivector_top20_docs, gt_key
        )
        multivector_top20_hit = bool(multivector_top20_correct_count > 0)

        if multivector_top1_correct:
            multivector_top1_correct_all += 1
            if eligible:
                multivector_top1_correct_eligible += 1
        if multivector_topk_hit:
            multivector_topk_hit_all += 1
            if eligible:
                multivector_topk_hit_eligible += 1
        if multivector_top20_hit:
            multivector_top20_hit_all += 1
            if eligible:
                multivector_top20_hit_eligible += 1
        multivector_topk_correct_total_all += int(multivector_topk_correct_count)
        if eligible:
            multivector_topk_correct_total_eligible += int(multivector_topk_correct_count)

        if eligible:
            if single_top1_correct and multivector_top1_correct:
                both_top1_correct += 1
            elif single_top1_correct and not multivector_top1_correct:
                single_only_top1_correct += 1
            elif multivector_top1_correct and not single_top1_correct:
                multivector_only_top1_correct += 1
            else:
                both_top1_wrong += 1

            if single_topk_hit and multivector_topk_hit:
                both_topk_hit += 1
            elif single_topk_hit and not multivector_topk_hit:
                single_only_topk_hit += 1
            elif multivector_topk_hit and not single_topk_hit:
                multivector_only_topk_hit += 1
            else:
                both_topk_miss += 1

        per_sample.append(
            {
                "sample_index": int(sample_index),
                "id_set": int(id_set),
                "eligible": bool(eligible),
                "single_top1_correct": bool(single_top1_correct),
                "single_top1_score": None if single_top1_score is None else float(single_top1_score),
                "single_top1_embedding_id": single_top1_embedding_id,
                "single_top1_id_set": single_top1_id_set,
                "single_top1_prompt": single_top1_prompt,
                "single_topk_count": int(len(single_topk_docs)),
                "single_topk_correct_count": int(single_topk_correct_count),
                "single_topk_hit": bool(single_topk_hit),
                "single_top20_hit": bool(single_top20_hit),
                "multivector_candidate_count": int(multivector_candidate_count),
                "multivector_top1_correct": bool(multivector_top1_correct),
                "multivector_top1_score": None if multivector_top1_score is None else float(multivector_top1_score),
                "multivector_top1_embedding_id": multivector_top1_embedding_id,
                "multivector_top1_id_set": multivector_top1_id_set,
                "multivector_top1_prompt": multivector_top1_prompt,
                "multivector_topk_count": int(len(multivector_topk_docs)),
                "multivector_topk_correct_count": int(multivector_topk_correct_count),
                "multivector_topk_hit": bool(multivector_topk_hit),
                "multivector_top20_hit": bool(multivector_top20_hit),
                "prompt": prompt,
            }
        )

        t0 = time.time()
        embedding_id = int(next_embedding_id)
        next_embedding_id += 1
        insert_tensor = _build_chunked_maxsim_tensor(
            splitter=splitter,
            text=prompt,
            chunk_token_limit=int(args.chunk_token_limit),
            timing=timing,
            timing_prefix="insert_",
        )
        insert_vectors_np = insert_tensor.detach().float().cpu().numpy()
        doc = IndexedDoc(
            embedding_id=int(embedding_id),
            prompt=prompt,
            id_set=int(id_set),
            gt_key=gt_key,
            pooled_knn=pooled_knn_np,
            pooled_no_cls=query_enc["pooled_no_cls"].detach(),
            maxsim_tensor=insert_tensor.detach(),
        )
        docs_by_id[int(embedding_id)] = doc
        multivector_index.add_parent_vectors(
            parent_id=int(embedding_id),
            vectors=insert_vectors_np,
        )
        timing.add("insert_s", time.time() - t0)

        if gt_key:
            seen_gt_key_counts[gt_key] = seen_gt_key_counts.get(gt_key, 0) + 1

        pbar.set_description(
            "Comparing retrieval top-k "
            f"single={single_top1_correct_all}/{sample_index} "
            f"multi={multivector_top1_correct_all}/{sample_index}"
        )
        pbar.set_postfix(
            eligible=int(eligible_samples),
            s_r1=f"{_safe_rate(single_top1_correct_eligible, eligible_samples):.3f}",
            s_r20=f"{_safe_rate(single_top20_hit_eligible, eligible_samples):.3f}",
            m_r1=f"{_safe_rate(multivector_top1_correct_eligible, eligible_samples):.3f}",
            m_r20=f"{_safe_rate(multivector_top20_hit_eligible, eligible_samples):.3f}",
        )

    total_samples = len(per_sample)
    avg_multivector_candidate_count = _safe_rate(
        total_multivector_candidate_count, total_samples
    )
    avg_multivector_rerank_count = _safe_rate(total_multivector_rerank_count, total_samples)

    summary = {
        "total_samples": int(total_samples),
        "eligible_samples": int(eligible_samples),
        "single": {
            "top1_accuracy_all": _safe_rate(single_top1_correct_all, total_samples),
            "top1_accuracy_eligible": _safe_rate(single_top1_correct_eligible, eligible_samples),
            "top20_recall_all": _safe_rate(single_top20_hit_all, total_samples),
            "top20_recall_eligible": _safe_rate(single_top20_hit_eligible, eligible_samples),
            "topk_hit_rate_all": _safe_rate(single_topk_hit_all, total_samples),
            "topk_hit_rate_eligible": _safe_rate(single_topk_hit_eligible, eligible_samples),
            "avg_correct_in_topk_all": _safe_rate(single_topk_correct_total_all, total_samples),
            "avg_correct_in_topk_eligible": _safe_rate(
                single_topk_correct_total_eligible, eligible_samples
            ),
        },
        "multivector": {
            "top1_accuracy_all": _safe_rate(multivector_top1_correct_all, total_samples),
            "top1_accuracy_eligible": _safe_rate(
                multivector_top1_correct_eligible, eligible_samples
            ),
            "top20_recall_all": _safe_rate(multivector_top20_hit_all, total_samples),
            "top20_recall_eligible": _safe_rate(
                multivector_top20_hit_eligible, eligible_samples
            ),
            "topk_hit_rate_all": _safe_rate(multivector_topk_hit_all, total_samples),
            "topk_hit_rate_eligible": _safe_rate(
                multivector_topk_hit_eligible, eligible_samples
            ),
            "avg_correct_in_topk_all": _safe_rate(
                multivector_topk_correct_total_all, total_samples
            ),
            "avg_correct_in_topk_eligible": _safe_rate(
                multivector_topk_correct_total_eligible, eligible_samples
            ),
            "avg_candidates_before_rerank": float(avg_multivector_candidate_count),
            "avg_candidates_after_rerank": float(avg_multivector_rerank_count),
        },
        "pairwise_eligible": {
            "top1": {
                "both_correct": int(both_top1_correct),
                "single_only_correct": int(single_only_top1_correct),
                "multivector_only_correct": int(multivector_only_top1_correct),
                "both_wrong": int(both_top1_wrong),
            },
            "topk_hit": {
                "both_hit": int(both_topk_hit),
                "single_only_hit": int(single_only_topk_hit),
                "multivector_only_hit": int(multivector_only_topk_hit),
                "both_miss": int(both_topk_miss),
            },
        },
        "timing": {
            "encode_s": float(timing.get("encode_s")),
            "single_full_scan_s": float(timing.get("single_full_scan_s")),
            "multi_query_tensor_s": float(timing.get("multi_query_tensor_s")),
            "multi_retrieval_s": float(timing.get("multi_retrieval_s")),
            "multi_rerank_s": float(timing.get("multi_rerank_s")),
            "query_chunk_encode_s": float(timing.get("query_chunk_encode_s")),
            "query_chunk_segment_s": float(timing.get("query_chunk_segment_s")),
            "insert_chunk_encode_s": float(timing.get("insert_chunk_encode_s")),
            "insert_chunk_segment_s": float(timing.get("insert_chunk_segment_s")),
            "insert_s": float(timing.get("insert_s")),
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
            "eval_top_k": int(args.eval_top_k),
            "chunk_token_limit": int(args.chunk_token_limit),
            "multivector_max_elements": int(args.multivector_max_elements),
            "ground_truth_mode": str(ground_truth_mode),
            "llm_col": args.llm_col,
        },
        "summary": summary,
        "per_sample": per_sample,
    }

    print(f"dataset={args.dataset}")
    print(f"total_samples={total_samples}")
    print(f"eligible_samples={eligible_samples}")
    print(
        "running recall targets: top1 and top20 (eligible-based live progress)"
    )
    print(
        "single recall@1 all={:.4f} eligible={:.4f} | recall@20 all={:.4f} eligible={:.4f} | topk_hit all={:.4f} eligible={:.4f} | avg_correct_topk all={:.4f}".format(
            summary["single"]["top1_accuracy_all"],
            summary["single"]["top1_accuracy_eligible"],
            summary["single"]["top20_recall_all"],
            summary["single"]["top20_recall_eligible"],
            summary["single"]["topk_hit_rate_all"],
            summary["single"]["topk_hit_rate_eligible"],
            summary["single"]["avg_correct_in_topk_all"],
        )
    )
    print(
        "multi recall@1 all={:.4f} eligible={:.4f} | recall@20 all={:.4f} eligible={:.4f} | topk_hit all={:.4f} eligible={:.4f} | avg_correct_topk all={:.4f}".format(
            summary["multivector"]["top1_accuracy_all"],
            summary["multivector"]["top1_accuracy_eligible"],
            summary["multivector"]["top20_recall_all"],
            summary["multivector"]["top20_recall_eligible"],
            summary["multivector"]["topk_hit_rate_all"],
            summary["multivector"]["topk_hit_rate_eligible"],
            summary["multivector"]["avg_correct_in_topk_all"],
        )
    )
    print(
        "pairwise top1 eligible: both_correct={both} single_only={single_only} multivector_only={multi_only} both_wrong={wrong}".format(
            both=both_top1_correct,
            single_only=single_only_top1_correct,
            multi_only=multivector_only_top1_correct,
            wrong=both_top1_wrong,
        )
    )
    print(
        "pairwise topk eligible: both_hit={both} single_only={single_only} multivector_only={multi_only} both_miss={miss}".format(
            both=both_topk_hit,
            single_only=single_only_topk_hit,
            multi_only=multivector_only_topk_hit,
            miss=both_topk_miss,
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
