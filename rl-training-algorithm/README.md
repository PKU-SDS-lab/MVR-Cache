# RL Training Algorithm

This directory contains the reinforcement-learning training code for the learned prompt segmentation model used by **MVR-cache**.

The main entry point is:

```bash
python RL4COTrainer.py
```

## What This Code Does

The training pipeline learns a segmentation policy that splits a pair of prompts into segments so that MaxSim-based similarity scoring can better identify semantically matching prompts under correctness constraints.

Core files:

- `RL4COTrainer.py`: main training script
- `MaxSimEnv.py`: RL environment and reward computation
- `MaxSimGenerator.py`: prompt-pair generator and sampling logic
- `AdaptedPointerNetworkPolicy.py`: segmentation policy
- `calibrator.py`: calibration module used in training
- `similarity_evaluator.py`: label construction helpers for parquet-based datasets

## Input Data

The trainer supports three input styles:

- pairwise JSON data via `--train_pairs_json`, `--val_pairs_json`, `--test_pairs_json`
- raw prompt text files via `--train_file`, `--val_file`, `--test_file`
- parquet datasets via `--train_parquet`, `--val_parquet`, `--test_parquet`

For the paper setup, parquet training was used together with semantic labels derived from dataset metadata.

Expected parquet structure:

- required: `prompt`
- optional for label construction: `ID_Set`, `id_set`, or `label_id`
- optional for string-based labels: a response column such as `response_gpt-4o-mini`

## Example: Paper-Style Training Command

```bash
cd rl-training-algorithm

python RL4COTrainer.py \
  --gpu_id 0 \
  --train_parquet /path/to/train_10k.parquet \
  --val_parquet /path/to/val_1k.parquet \
  --test_parquet /path/to/test_1k.parquet \
  --parquet_text_column prompt \
  --label_mode id_set \
  --train_sampling_mode anchor_nn \
  --nn_warmup_epochs 5 \
  --nn_candidate_topk 10 \
  --nn_rebuild_every_n_epochs 1 \
  --train_data_size 10000 \
  --batch_size 8 \
  --accumulate_grad_batches 2 \
  --lr 1e-4 \
  --max_epochs 200 \
  --check_val_every_n_epoch 5 \
  --checkpoint_dir ./checkpoints/sq_10k_ckpt \
  --policy_mode separate \
  --punctuation_only \
  --bce_auto_balance \
  --precompute_token_embeddings
```

## Important Arguments

- `--checkpoint_dir`: directory where checkpoints are saved
- `--policy_mode separate`: the mode used in the paper experiments
- `--train_sampling_mode anchor_nn`: nearest-neighbor guided sampling
- `--label_mode id_set`: uses `ID_Set` or `id_set` equality as supervision
- `--punctuation_only`: restricts split points to punctuation tokens
- `--precompute_token_embeddings`: useful for speeding up repeated training batches
- `--bce_auto_balance`: auto-balances BCE terms under class imbalance

## Output

Training writes checkpoints under `--checkpoint_dir`. The evaluation code in `mvr-cache` accepts either:

- a specific checkpoint file, or
- a checkpoint directory containing files like `epoch=...-step=....ckpt`

The MVR-cache evaluation code resolves the newest checkpoint automatically when you pass a directory.

## Utilities

Helpful scripts:

- `scripts/checkpoint_smoketest.py`: quick checkpoint sanity check
- `scripts/eval_testset_report.py`: post-training evaluation helper
- `scripts/single_text_inference.py`: inspect segmentation behavior on example prompts
- `make_balanced_pairs.py`: pair construction utility

## Notes

- The code expects `rl4co`, `torchrl`, `lightning`, `transformers`, and `pyarrow`.
- Set `BGE_MODEL_PATH` if you want to force a local encoder path.
- Checkpoints are not included in this release.
