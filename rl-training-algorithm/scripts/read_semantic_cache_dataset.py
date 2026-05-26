#!/usr/bin/env python3
"""
Read and inspect the semantic cache dataset JSON.

Example:
  python scripts/read_semantic_cache_dataset.py \
    --path /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_benchmark200SemanticChunk.json \
    --head 3
"""

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _safe_len(x: Any) -> Optional[int]:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _summarize_records(records: List[Dict[str, Any]], sample_n: int = 200) -> Dict[str, Any]:
    n = len(records)
    keys_ctr: Counter[str] = Counter()
    task_ctr: Counter[str] = Counter()
    answer_ctr: Counter[str] = Counter()
    chunk_lens: List[int] = []

    for r in records[: min(n, sample_n)]:
        if isinstance(r, dict):
            keys_ctr.update(r.keys())
            if "task" in r and isinstance(r["task"], str):
                task_ctr.update([r["task"]])
            if "answer" in r and isinstance(r["answer"], str):
                answer_ctr.update([r["answer"]])
            if "SemanticChunk" in r and isinstance(r["SemanticChunk"], list):
                chunk_lens.append(len(r["SemanticChunk"]))

    out: Dict[str, Any] = {
        "n_records": n,
        "sampled": min(n, sample_n),
        "top_keys": keys_ctr.most_common(30),
        "tasks": task_ctr.most_common(20),
        "answers": answer_ctr.most_common(20),
    }
    if chunk_lens:
        out["semantic_chunk_len"] = {
            "min": min(chunk_lens),
            "max": max(chunk_lens),
            "avg": sum(chunk_lens) / len(chunk_lens),
        }
    return out


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to the .json dataset file")
    ap.add_argument("--head", type=int, default=3, help="Print first N records (default: 3)")
    ap.add_argument(
        "--summary_sample",
        type=int,
        default=200,
        help="How many records to scan for summary stats (default: 200)",
    )
    args = ap.parse_args()

    data = load_json(args.path)

    # The file you pointed to is a JSON array of objects.
    if isinstance(data, list):
        print(f"[OK] Loaded JSON list with {len(data)} records from: {args.path}")
        if data and isinstance(data[0], dict):
            print("[INFO] First record keys:", sorted(list(data[0].keys())))
        print()

        print("[SUMMARY]")
        summary = _summarize_records(data, sample_n=args.summary_sample)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print()

        if args.head > 0:
            print(f"[HEAD] First {min(args.head, len(data))} records:")
            for i, rec in enumerate(data[: args.head]):
                # Make output readable; don’t dump huge text fields fully
                if isinstance(rec, dict):
                    rec2 = dict(rec)
                    if "sentence" in rec2 and isinstance(rec2["sentence"], str) and len(rec2["sentence"]) > 300:
                        rec2["sentence"] = rec2["sentence"][:300] + " ... (truncated)"
                    if "SemanticChunk" in rec2 and isinstance(rec2["SemanticChunk"], list) and len(rec2["SemanticChunk"]) > 5:
                        rec2["SemanticChunk"] = rec2["SemanticChunk"][:5] + ["... (truncated)"]
                    print(f"\n--- record[{i}] ---")
                    print(json.dumps(rec2, indent=2, ensure_ascii=False))
                else:
                    print(f"\n--- record[{i}] (non-dict) ---")
                    print(rec)
    elif isinstance(data, dict):
        print(f"[OK] Loaded JSON object from: {args.path}")
        print("[INFO] Top-level keys:", sorted(list(data.keys())))
        print("[INFO] Top-level key lengths:", {k: _safe_len(v) for k, v in data.items()})
    else:
        print(f"[OK] Loaded JSON value type={type(data)} from: {args.path}")
        print(data)


if __name__ == "__main__":
    main()


