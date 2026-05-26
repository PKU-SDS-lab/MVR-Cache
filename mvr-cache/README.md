# MVR-cache and vCache Baselines

This directory contains:

- the original **vCache** baseline code path
- the **MVR-cache** variant described in the paper
- benchmark scripts for both baseline and splitter-based evaluation

The main public entry points are in [`benchmarks/`](./benchmarks).

## Installation

From this directory:

```bash
cd mvr-cache
python -m pip install -e .
python -m pip install -e vcache/vcache_core/cache/embedding_store/hnswlib
```

The second command installs the local modified HNSW backend required for multi-vector retrieval.

## What to Run

### 1. vCache Baseline

Use [`benchmarks/eval_sembenchmark_verified.py`](./benchmarks/eval_sembenchmark_verified.py).

Example:

```bash
python benchmarks/eval_sembenchmark_verified.py \
  --dataset /path/to/dataset.csv \
  --llm-col response_llama_3_8b \
  --delta 0.01 \
  --similarity-evaluator benchmark_id_set \
  --sleep 0.02 \
  --device cuda \
  --output-json results/baseline_verified.json
```

### 2. MVR-cache

Use [`benchmarks/eval_sembenchmark_verified_splitter.py`](./benchmarks/eval_sembenchmark_verified_splitter.py).

Example:

```bash
python benchmarks/eval_sembenchmark_verified_splitter.py \
  --dataset /path/to/dataset.csv \
  --deltas 0.01 0.015 0.02 0.03 0.05 0.07 0.08 \
  --candidate-selection multivector_top_k \
  --candidate-k 10 \
  --splitter-checkpoint ../rl-training-algorithm/checkpoints/sq_10k_ckpt \
  --splitter-device cuda:0 \
  --mix-fullcos \
  --include-full-embedding \
  --splitter-overlap-tokens 0 \
  --similarity-evaluator benchmark_id_set \
  --sleep 0.1 \
  --output-json results/mvr_cache.json \
  --benchmark-output-dir results/benchmark_runs \
  --benchmark-run-index 1 \
  --save-cache-hit-samples results/cache_hit_samples.jsonl
```

### 3. Insert-All Variants

If you want the insert-all versions used in some experiments:

- baseline insert-all: `benchmarks/eval_sembenchmark_verified_insert_all.py`
- MVR-cache insert-all: `benchmarks/eval_sembenchmark_verified_splitter_insert_all.py`

## Dataset Format

For local CSV or parquet files:

- required: `prompt`
- optional: `output_format`
- label columns for `benchmark_id_set` mode: `ID_Set`, `id_set`, or `label_id`
- response column for `string` mode: pass it through `--llm-col`

The scripts also support Hugging Face datasets such as:

- `vCache/SemBenchmarkClassification`
- `vCache/SemBenchmarkLmArena`
- `vCache/SemBenchmarkSearchQueries`

## Which Evaluator To Use

- `--similarity-evaluator benchmark_id_set`: use ID-set labels from the dataset
- `--similarity-evaluator string`: compare responses using the column named by `--llm-col`

For the paper-style datasets, `benchmark_id_set` is the cleanest option when the dataset provides label IDs.

## Candidate Selection Modes

- `top_k`: retrieve top-k candidates using single-vector retrieval
- `all`: scan all cached prompts
- `multivector_top_k`: use the modified multi-vector HNSW backend

For **MVR-cache** as described in the paper, use:

```text
--candidate-selection multivector_top_k
--candidate-k 10
```

## Notes on `--sleep`

The verified policies update internal state asynchronously. In these benchmark scripts, `--sleep` is used as a synchronization workaround so background updates complete correctly during offline evaluation. Keep it enabled for reproducibility, but do not treat this artificial sleep as part of the method latency.

## Outputs

Depending on the script and arguments, outputs include:

- per-run JSON summaries
- aggregate JSON summaries for multiple deltas
- cache-hit sample JSONL files
- benchmark-compatible result directories for plotting

This release does not include any precomputed results.

## Related Files

- `benchmarks/benchmark.py`: broader benchmark driver
- `benchmarks/eval_segmentation_stats.py`: segmentation analysis
- `benchmarks/debug_splitter_segments.py`: inspect segment boundaries from a checkpoint
- `playground/`: small usage examples

## License

This directory preserves the upstream `vCache` license file that was present in the source tree used for this project. Review it before public redistribution.
