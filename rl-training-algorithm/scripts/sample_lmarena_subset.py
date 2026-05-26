"""
Sample a random subset of rows from an LMArena Parquet file and write it back out.

This keeps *all columns* (so you can use other fields later), and is reproducible via --seed.

Example:
  /data1/conda_envs/RLSemanticCaching/bin/python scripts/sample_lmarena_subset.py \
    --input_parquet /data2/ali/LMArena/train.parquet \
    --output_parquet /data2/ali/LMArena/train_10k.parquet \
    --output_csv /data2/ali/LMArena/train_10k.csv \
    --n 10000 \
    --seed 42
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", type=str, required=True)
    ap.add_argument("--output_parquet", type=str, required=True)
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.input_parquet):
        raise FileNotFoundError(args.input_parquet)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to sample parquet. "
            "Run with the conda env that has pyarrow installed."
        ) from e

    pf = pq.ParquetFile(args.input_parquet)
    n_rows = int(pf.metadata.num_rows)
    n = int(args.n)
    if n <= 0:
        raise ValueError("--n must be > 0")
    if n > n_rows:
        raise ValueError(f"--n={n} exceeds num_rows={n_rows}")

    # IMPORTANT:
    # The source parquet may contain very large string columns (embeddings / responses).
    # Materializing the full table and calling `Table.take(...)` can trigger
    # "offset overflow while concatenating arrays" even if you're only taking 10k rows,
    # because pyarrow may try to concatenate full-chunked string arrays first.
    #
    # To avoid this, we stream batches and take rows *within each RecordBatch*.
    import numpy as np

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_rows, size=n, replace=False)).astype(np.int64)

    batches_out = []
    cursor = 0
    row_base = 0
    # Tune batch_size if needed; larger is faster, smaller uses less peak memory.
    for batch in pf.iter_batches(batch_size=2048):
        bsz = batch.num_rows
        if cursor >= len(idx):
            break
        # Find all target indices that fall inside this batch's global row range
        start = row_base
        end = row_base + bsz
        # Advance cursor to first idx >= start (it should already be, but keep robust)
        while cursor < len(idx) and idx[cursor] < start:
            cursor += 1
        if cursor >= len(idx):
            break
        j = cursor
        while j < len(idx) and idx[j] < end:
            j += 1
        if j > cursor:
            local = (idx[cursor:j] - start).tolist()
            taken = batch.take(pa.array(local, type=pa.int64()))
            batches_out.append(taken)
            cursor = j
        row_base = end

    if not batches_out:
        raise RuntimeError("No rows were selected; check sampling logic.")

    subset = pa.Table.from_batches(batches_out)

    os.makedirs(os.path.dirname(args.output_parquet) or ".", exist_ok=True)
    pq.write_table(subset, args.output_parquet, compression="zstd")
    print(f"Wrote subset parquet: {args.output_parquet} rows={subset.num_rows}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        # CSV can be large; still fine for 10k rows.
        import pyarrow.csv as pacsv

        pacsv.write_csv(subset, args.output_csv)
        print(f"Wrote subset csv: {args.output_csv} rows={subset.num_rows}")


if __name__ == "__main__":
    main()

