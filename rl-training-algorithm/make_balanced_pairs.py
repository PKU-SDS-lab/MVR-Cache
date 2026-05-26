#!/usr/bin/env python3
"""
Create a balanced dataset of sentence pairs from a labeled dataset.

Input format: list[dict] where each dict has at least:
  - sentence (str)
  - answer (str)   # label

Output format: list[dict] with:
  - sentence_1 (str)
  - sentence_2 (str)
  - correct (int)  # 1 if same label else 0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pair_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i <= j else (j, i)


def build_pairs(
    sentences: List[str],
    labels: List[str],
    n_pairs: int,
    seed: int,
    max_attempts: int = 2_000_000,
) -> List[Dict[str, Any]]:
    if len(sentences) != len(labels):
        raise ValueError("sentences and labels must have the same length")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be > 0")
    if n_pairs % 2 != 0:
        raise ValueError("n_pairs must be even to balance 1/0")

    rng = Random(seed)

    idx_by_label: Dict[str, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        idx_by_label[lab].append(i)

    label_list = [lab for lab, idxs in idx_by_label.items() if len(idxs) >= 2]
    if not label_list:
        raise ValueError("Need at least one label with >= 2 samples to form positive pairs")

    all_labels = list(idx_by_label.keys())
    if len(all_labels) < 2:
        raise ValueError("Need at least 2 distinct labels to form negative pairs")

    n_pos = n_pairs // 2
    n_neg = n_pairs // 2

    used_pos = set()
    used_neg = set()
    out: List[Dict[str, Any]] = []

    # --- Positive pairs (same label) ---
    attempts = 0
    while len(used_pos) < n_pos:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Failed to sample {n_pos} positive pairs (attempts={attempts})")

        lab = rng.choice(label_list)
        idxs = idx_by_label[lab]
        i = rng.choice(idxs)
        j = rng.choice(idxs)
        if i == j:
            continue
        k = _pair_key(i, j)
        if k in used_pos:
            continue
        used_pos.add(k)

    # --- Negative pairs (different labels) ---
    attempts = 0
    while len(used_neg) < n_neg:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Failed to sample {n_neg} negative pairs (attempts={attempts})")

        lab1 = rng.choice(all_labels)
        lab2 = rng.choice(all_labels)
        if lab1 == lab2:
            continue
        i = rng.choice(idx_by_label[lab1])
        j = rng.choice(idx_by_label[lab2])
        if i == j:
            continue
        k = _pair_key(i, j)
        if k in used_neg:
            continue
        used_neg.add(k)

    # Materialize examples and shuffle so labels are mixed
    for (i, j) in used_pos:
        out.append({"sentence_1": sentences[i], "sentence_2": sentences[j], "correct": 1})
    for (i, j) in used_neg:
        out.append({"sentence_1": sentences[i], "sentence_2": sentences[j], "correct": 0})

    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_benchmark2500.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON file path (default: alongside input, with _pairs_balanced suffix)",
    )
    parser.add_argument(
        "--make_splits",
        action="store_true",
        help="If set, write train/val/test split files instead of a single output file",
    )
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction (splits mode)")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Val fraction (splits mode)")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction (splits mode)")
    parser.add_argument(
        "--min_punct",
        type=int,
        default=1,
        help="Only use source sentences that contain at least this many punctuation chars (set by --punct_chars). 0 disables filtering.",
    )
    parser.add_argument(
        "--punct_chars",
        type=str,
        default=",.!?:;，。！？：；",
        help="Characters counted as punctuation for filtering (default matches AdaptedPointerNetworkPolicy punct_chars).",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=0,
        help="Total number of pairs to generate (must be even). Default: len(input) if even else len(input)-1",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    data = _load_json(in_path)
    if not isinstance(data, list) or not data:
        raise ValueError("Input JSON must be a non-empty list")

    sentences: List[str] = []
    labels: List[str] = []
    punct_set = set(args.punct_chars)
    kept = 0
    dropped = 0
    for ex in data:
        if not isinstance(ex, dict):
            raise ValueError("Each example must be a dict")
        s = ex.get("sentence")
        a = ex.get("answer")
        if not isinstance(s, str) or not isinstance(a, str):
            raise ValueError("Each example must contain string fields: sentence, answer")
        if args.min_punct and args.min_punct > 0:
            pc = sum(1 for ch in s if ch in punct_set)
            if pc < args.min_punct:
                dropped += 1
                continue
        kept += 1
        sentences.append(s)
        labels.append(a)

    if args.min_punct and args.min_punct > 0:
        print(
            f"[punct-filter] min_punct={args.min_punct} punct_chars={args.punct_chars!r} "
            f"kept={kept} dropped={dropped}"
        )
        if kept < 10:
            raise ValueError("Too few examples after punctuation filtering; lower --min_punct or adjust --punct_chars")

    if args.num_pairs and args.num_pairs > 0:
        n_pairs = args.num_pairs
    else:
        n_pairs = len(sentences) if len(sentences) % 2 == 0 else (len(sentences) - 1)

    pairs = build_pairs(sentences, labels, n_pairs=n_pairs, seed=args.seed)

    # If splitting, we ensure each split is balanced by allocating positives/negatives separately.
    if args.make_splits:
        tf, vf, tef = float(args.train_frac), float(args.val_frac), float(args.test_frac)
        if tf <= 0 or vf < 0 or tef < 0 or abs((tf + vf + tef) - 1.0) > 1e-6:
            raise ValueError("Splits must be non-negative and sum to 1.0 (train_frac + val_frac + test_frac)")

        n_train = int(round(n_pairs * tf))
        n_val = int(round(n_pairs * vf))
        n_test = n_pairs - n_train - n_val

        # Make each split size even (required for balance); adjust by moving 1 pair at a time.
        def _make_even(x: int) -> int:
            return x if x % 2 == 0 else (x - 1)

        n_train = _make_even(n_train)
        n_val = _make_even(n_val)
        n_test = n_pairs - n_train - n_val
        n_test = _make_even(n_test)
        # final fix to keep sum == n_pairs
        total = n_train + n_val + n_test
        if total != n_pairs:
            # adjust train to absorb the difference (should be even)
            diff = n_pairs - total
            n_train += diff
            if n_train < 0 or n_train % 2 != 0:
                raise RuntimeError("Failed to compute even split sizes")

        pos = [x for x in pairs if x["correct"] == 1]
        neg = [x for x in pairs if x["correct"] == 0]
        assert len(pos) == len(neg) == n_pairs // 2

        # Split counts per class
        train_pos = n_train // 2
        val_pos = n_val // 2
        test_pos = n_test // 2

        train = pos[:train_pos] + neg[:train_pos]
        val = pos[train_pos : train_pos + val_pos] + neg[train_pos : train_pos + val_pos]
        test = pos[train_pos + val_pos : train_pos + val_pos + test_pos] + neg[train_pos + val_pos : train_pos + val_pos + test_pos]

        # Shuffle each split deterministically
        rng = Random(args.seed)
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        base = in_path.with_name(in_path.stem + "_pairs_balanced")
        out_train = base.with_name(base.name + "_train.json")
        out_val = base.with_name(base.name + "_val.json")
        out_test = base.with_name(base.name + "_test.json")
        _dump_json(out_train, train)
        _dump_json(out_val, val)
        _dump_json(out_test, test)

        def _summ(name: str, xs: List[Dict[str, Any]]) -> None:
            n1 = sum(1 for x in xs if x["correct"] == 1)
            n0 = sum(1 for x in xs if x["correct"] == 0)
            print(f"{name}: {len(xs)} (correct=1: {n1}, correct=0: {n0})")

        print("Wrote split files:")
        print(f"  {out_train}")
        print(f"  {out_val}")
        print(f"  {out_test}")
        _summ("train", train)
        _summ("val", val)
        _summ("test", test)
        return

    # Single-file output
    if not args.output:
        out_path = in_path.with_name(in_path.stem + "_pairs_balanced.json")
    else:
        out_path = Path(args.output)
    _dump_json(out_path, pairs)

    n1 = sum(1 for x in pairs if x["correct"] == 1)
    n0 = sum(1 for x in pairs if x["correct"] == 0)
    print(f"Wrote: {out_path}")
    print(f"Total pairs: {len(pairs)} | correct=1: {n1} | correct=0: {n0}")
    print("First 2 examples:")
    for ex in pairs[:2]:
        print(
            {
                "correct": ex["correct"],
                "sentence_1_preview": ex["sentence_1"][:120],
                "sentence_2_preview": ex["sentence_2"][:120],
            }
        )


if __name__ == "__main__":
    main()


