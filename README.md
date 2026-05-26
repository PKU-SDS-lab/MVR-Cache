# MVR-cache

Code release for the paper **MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation** by **Ali Noshad, Zishan Zheng, and Yinjun Wu**. The paper was published in **ICML 2026** and is available at [`arXiv:2605.24914`](https://arxiv.org/abs/2605.24914).

This release is organized into two code paths:

- `rl-training-algorithm/`: reinforcement-learning training code for the learned prompt segmentation model.
- `mvr-cache/`: semantic caching and evaluation code, including both the original vCache baseline and the MVR-cache variant that uses the learned splitter together with multi-vector retrieval.

The release intentionally excludes:

- trained checkpoints and model weights
- benchmark result files
- generated plots
- local logs and scratch outputs

## Repository Layout

```text
paper_release/
├── rl-training-algorithm/
├── mvr-cache/
├── environment/
└── .gitignore
```

## Environment

The original development environment used the conda environment name `RLSemanticCaching`. Two environment manifests are included in [`environment/`](./environment):

- `RLSemanticCaching.yml`: full environment export
- `RLSemanticCaching.from-history.yml`: minimal history-based export

For a fresh setup, start from the full export or recreate the environment manually.

Example:

```bash
conda env create -f environment/RLSemanticCaching.yml
conda activate RLSemanticCaching
```

If you prefer a lighter setup, the cache package also includes a Poetry manifest in [`mvr-cache/pyproject.toml`](./mvr-cache/pyproject.toml), but the conda environment is the closest match to the environment used for the paper experiments.

## Model Assets and Hugging Face Cache

Both training and evaluation use a BGE encoder. The code will:

1. use `BGE_MODEL_PATH` if you set it
2. otherwise try a small repo-local cache if you provide one
3. otherwise download `BAAI/bge-base-en-v1.5`

Recommended environment variables:

```bash
export BGE_MODEL_PATH=/path/to/local/BAAI__bge-base-en-v1.5
export HF_HOME=/path/to/hf/home
export HF_HUB_CACHE=/path/to/hf/hub
export DATASETS_CACHE=/path/to/hf/datasets
export TRANSFORMERS_CACHE=/path/to/hf/transformers
```

If you need to use a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Typical Workflow

1. Train the learned segmentation model in `rl-training-algorithm/`.
2. Save checkpoints to a directory such as `checkpoints/sq_10k_ckpt/`.
3. Run baseline vCache evaluations from `mvr-cache/benchmarks/eval_sembenchmark_verified.py`.
4. Run MVR-cache evaluations from `mvr-cache/benchmarks/eval_sembenchmark_verified_splitter.py`, pointing `--splitter-checkpoint` to the trained checkpoint directory.

## Running the RL Training

See [`rl-training-algorithm/README.md`](./rl-training-algorithm/README.md).

## Running vCache and MVR-cache

See [`mvr-cache/README.md`](./mvr-cache/README.md).

## Notes on the Multi-Vector HNSW Backend

The `mvr-cache` package vendors the modified HNSW implementation used for multi-vector retrieval under:

[`mvr-cache/vcache/vcache_core/cache/embedding_store/hnswlib`](./mvr-cache/vcache/vcache_core/cache/embedding_store/hnswlib)

To use the multi-vector retrieval path, install that local package inside the active environment:

```bash
cd mvr-cache
python -m pip install -e vcache/vcache_core/cache/embedding_store/hnswlib
```

This is required for `--candidate-selection multivector_top_k`.

## Citation

```bibtex
@article{noshad2026mvrcache,
  title={MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation},
  author={Noshad, Ali and Zheng, Zishan and Wu, Yinjun},
  journal={arXiv preprint arXiv:2605.24914},
  year={2026}
}
```

## License

The `mvr-cache/` directory preserves the upstream `vCache` license file included in the source it extends. Verify that this license matches your intended public redistribution terms before publishing the repository.
