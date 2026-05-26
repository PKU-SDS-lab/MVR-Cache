import os

# Respect an existing mirror configuration, but default to the public endpoint.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
# classification:  batch_size=24 TRAIN_DATA_SIZE=2560 lr=1e-4 epochs=200
#    poetry run  python ../RL_SemanticCashing_768/RL4COTrainer.py   --train_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_lmbenchmark2500_pairs_balanced_train.json   --val_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_lmbenchmark2500_pairs_balanced_val.json   --test_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_lmbenchmark2500_pairs_balanced_test.json   --checkpoint_dir ~/checkpoints_words_lm2   --policy_mode separate   --debug_policy   --debug_policy_log_path ~/checkpoints_words_lm2/policy_debug.jsonl   --debug_policy_every_n_epochs 1   --debug_policy_batch_size 8   --debug_policy_topk 10   --debug_policy_n_samples 2   --lr 5e-5 --batch_size 24  --accumulate_grad_batches 2
# poetry run  python ../RL_SemanticCashing_768/RL4COTrainer.py   --train_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_lmbenchmark2500_pairs_balanced_train.json   --val_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_lmbenchmark2500_pairs_balanced_val.json   --test_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_lmbenchmark2500_pairs_balanced_test.json   --checkpoint_dir ~/checkpoints_words_bef   --policy_mode separate   --debug_policy   --debug_policy_log_path ~/checkpoints_words_bef/policy_debug.jsonl   --debug_policy_every_n_epochs 1   --debug_policy_batch_size 8   --debug_policy_topk 10   --debug_policy_n_samples 2   --lr 5e-5 --batch_size 24  --accumulate_grad_batches 2 --split_words_before --gpu_id 1
import torch
import torch.nn.functional as F
import numpy as np
import argparse 
import json
import random
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from tensordict.tensordict import TensorDict
from rl4co.models.rl import REINFORCE
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks import Callback

from MaxSimEnv import MaxSimEnv
from MaxSimGenerator import MaxSimGenerator
from AdaptedPointerNetworkPolicy import AdaptedPointerNetworkPolicy
from embedding_model import EmbeddingModel
from calibrator import SigmoidCalibrator

def _count_parquet_rows(path: str) -> int:
    """Fast row count from parquet metadata (does not load full table)."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Need pyarrow to read parquet metadata for {path!r}. "
            "Run inside the conda env that has pyarrow installed."
        ) from e
    pf = pq.ParquetFile(path)
    return int(pf.metadata.num_rows)

def _infer_dataset_size(*, pairs, prompts, parquet_path: str | None) -> int:
    if pairs is not None:
        return len(pairs)
    if prompts is not None:
        return len(prompts)
    if parquet_path:
        return _count_parquet_rows(parquet_path)
    raise ValueError("Cannot infer dataset size: no pairs, no prompts, and no parquet_path.")

def _print_cuda_mem(tag):
    try:
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            name = torch.cuda.get_device_name(dev)
            alloc = torch.cuda.memory_allocated(dev) / (1024**2)
            reserved = torch.cuda.memory_reserved(dev) / (1024**2)
            max_alloc = torch.cuda.max_memory_allocated(dev) / (1024**2)
            print(
                f"[MEM][{tag}] device={dev}({name}) alloc={alloc:.1f}MB reserved={reserved:.1f}MB max_alloc={max_alloc:.1f}MB"
            )
        else:
            print(f"[MEM][{tag}] CUDA not available")
    except Exception as e:
        print(f"[MEM][{tag}] failed: {e}")

class ResumeFriendlyREINFORCE(REINFORCE):


    def __init__(self, *args, **kwargs):
        # What scalar reward REINFORCE optimizes:
        # - cal_only: reward = -BCE (loss-only reward; requires labels)
        # - signed_maxsim: reward = (+1 for correct, -1 for incorrect) * MaxSim
        # - signed_maxsim_plus_cal: reward = signed_maxsim + alpha * (-BCE)
        self.reward_mode = str(kwargs.pop("reward_mode", "cal_only")).lower().strip()
        if self.reward_mode not in {"cal_only", "signed_maxsim", "signed_maxsim_plus_cal"}:
            raise ValueError(
                f"Unsupported reward_mode={self.reward_mode!r}. Use cal_only|signed_maxsim|signed_maxsim_plus_cal."
            )
        self.alpha_cal_reward = kwargs.pop("alpha_cal_reward", 0.1)
        self.beta_cal_loss = kwargs.pop("beta_cal_loss", 1.0)
        # Class weights for BCE (to mitigate imbalance). These scale the per-example BCE.
        # If your batches are skewed heavily positive (as in anchor_nn), increase neg_weight.
        self.bce_pos_weight = float(kwargs.pop("bce_pos_weight", 1.0))
        self.bce_neg_weight = float(kwargs.pop("bce_neg_weight", 1.0))
        # If enabled, compute additional per-batch balancing weights from the observed label ratios.
        # This is useful when the sampler's label balance changes over time.
        self.bce_auto_balance = bool(kwargs.pop("bce_auto_balance", False))

        super().__init__(*args, **kwargs)

        # Similarity is normalized to [0, 1] (see MaxSimEnv reward mapping), so a natural
        # initial threshold is ~0.5.
        self.calibrator = SigmoidCalibrator(init_t=0.5, init_gamma=10.0)


    def on_load_checkpoint(self, checkpoint):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

            for key in state_dict:
                if isinstance(state_dict[key], torch.Tensor):
                    state_dict[key] = state_dict[key].clone()
        super().on_load_checkpoint(checkpoint)

    # NOTE: RL4CO/Lightning may call `setup()` with no args (e.g., during load_from_checkpoint),
    # so `stage` must be optional.
    def setup(self, stage: str | None = None):
        """Called at the beginning of fit, validate, test, or predict.
        Moves the embedding models to GPU after Lightning moves the model.
        """
        super().setup(stage)

        # Prefer the LightningModule device (what the Trainer actually uses).
        # Relying on policy.parameters().device can be misleading early in setup and can cause us to
        # accidentally move heavy submodules back to CPU (hurting speed).
        policy_device = getattr(self, "device", None)
        if policy_device is None:
            try:
                policy_device = next(self.policy.parameters()).device
            except StopIteration:
                policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEVICE] Module device: {policy_device}")

        # Move generator's embedding model to match the module device
        if hasattr(self, 'env') and hasattr(self.env, 'generator'):
            generator = self.env.generator
            if hasattr(generator, 'embedding_model') and generator.embedding_model is not None:
                current_device = next(generator.embedding_model.model.parameters()).device
                if current_device != policy_device:
                    generator.embedding_model.model.to(policy_device)
                    print(f"[DEVICE] Moved generator.embedding_model.model from {current_device} to {policy_device}")
                else:
                    print(f"[DEVICE] generator.embedding_model.model already on {policy_device}")

        # Also ensure the reward_lm in env is on the correct device
        if hasattr(self, 'env') and hasattr(self.env, 'reward_lm'):
            if hasattr(self.env.reward_lm, 'model'):
                current_device = next(self.env.reward_lm.model.parameters()).device
                if current_device != policy_device:
                    self.env.reward_lm.model.to(policy_device)
                    print(f"[DEVICE] Moved env.reward_lm.model from {current_device} to {policy_device}")
                else:
                    print(f"[DEVICE] env.reward_lm.model already on {policy_device}")

        # Move policy's embedding model if it exists
        if hasattr(self.policy, 'embedding_model') and self.policy.embedding_model is not None:
            current_device = next(self.policy.embedding_model.model.parameters()).device
            if current_device != policy_device:
                self.policy.embedding_model.model.to(policy_device)
                print(f"[DEVICE] Moved policy.embedding_model.model from {current_device} to {policy_device}")
            else:
                print(f"[DEVICE] policy.embedding_model.model already on {policy_device}")

    def _compute_shaped_reward_for_logging(self, td, base_reward_raw: torch.Tensor) -> torch.Tensor:
        """Compute shaped reward using the same calibration logic as training, but safe for val/test logging.

        - Does NOT backprop (uses detached tensors)
        - If labels (`td['correct']`) are unavailable, shaped reward == base reward
        """
        base_reward_flat = base_reward_raw.reshape(-1)  # [B]
        sim_for_cal = base_reward_flat  # raw similarity (can be [-1,1])

        # Default: no calibration shaping if labels missing
        r_cal = torch.zeros_like(sim_for_cal)

        if hasattr(td, "keys") and ("correct" in td.keys()):
            c = td["correct"].float().view(-1).to(sim_for_cal.device)  # [B]
            logits = self.calibrator.logits(sim_for_cal.detach())

            pos_w = float(getattr(self, "bce_pos_weight", 1.0))
            neg_w = float(getattr(self, "bce_neg_weight", 1.0))

            # Optional auto-balance weights based on batch label counts
            auto = bool(getattr(self, "bce_auto_balance", False))
            if auto:
                pos_cnt = (c >= 0.5).float().sum()
                neg_cnt = (c < 0.5).float().sum()
                if float(pos_cnt.detach().cpu().item()) > 0.0 and float(neg_cnt.detach().cpu().item()) > 0.0:
                    total = (pos_cnt + neg_cnt).detach()
                    auto_pos = (total / (2.0 * pos_cnt)).detach()
                    auto_neg = (total / (2.0 * neg_cnt)).detach()
                    pos_w = float((auto_pos * pos_w).detach().cpu().item())
                    neg_w = float((auto_neg * neg_w).detach().cpu().item())

            w = torch.where(
                c >= 0.5,
                torch.full_like(c, float(pos_w)),
                torch.full_like(c, float(neg_w)),
            )
            bce_per = F.binary_cross_entropy_with_logits(logits, c, weight=w, reduction="none")  # [B]
            r_cal = (-bce_per).detach()

        mode = str(getattr(self, "reward_mode", "cal_only")).lower().strip()
        # signed similarity: +sim for positives, -sim for negatives
        signed_sim = sim_for_cal
        if hasattr(td, "keys") and ("correct" in td.keys()):
            c01 = td["correct"].float().view(-1).to(sim_for_cal.device)  # [B]
            sign = (2.0 * (c01 >= 0.5).float() - 1.0)  # {+1,-1}
            signed_sim = sim_for_cal * sign

        if mode == "signed_maxsim":
            shaped_reward_flat = signed_sim
        elif mode == "signed_maxsim_plus_cal":
            alpha = float(getattr(self, "alpha_cal_reward", 0.5))
            shaped_reward_flat = signed_sim + alpha * r_cal
        else:
            # cal_only: if labels are missing, fall back to base similarity for logging
            if hasattr(td, "keys") and ("correct" in td.keys()):
                shaped_reward_flat = r_cal
            else:
                shaped_reward_flat = sim_for_cal
        return shaped_reward_flat.reshape_as(base_reward_raw)

    def shared_step(self, batch, batch_idx: int, phase: str, dataloader_idx: int = None):
        """Override RL4CO shared_step so validation metric uses shaped reward (for checkpointing)."""
        td = self.env.reset(batch)
        out = self.policy(td, self.env, phase=phase, select_best=phase != "train")

        if phase == "train":
            # Training uses the existing loss path (includes reward shaping for REINFORCE objective).
            out = self.calculate_loss(td, batch, out)
        else:
            # For val/test, compute shaped reward for logging/checkpoint selection (no backprop).
            base_reward = out.get("reward", None)
            if isinstance(base_reward, torch.Tensor):
                shaped_reward = self._compute_shaped_reward_for_logging(td, base_reward)
                out["reward"] = shaped_reward  # makes `val/reward` reflect shaped reward

                # Extra visibility in TensorBoard (optional)
                try:
                    self.log(
                        f"{phase}/reward_base",
                        base_reward.detach().mean(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                    self.log(
                        f"{phase}/reward_shaped",
                        shaped_reward.detach().mean(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                except Exception:
                    pass

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        td,
        batch,
        policy_out: dict,
        reward=None,
        log_likelihood=None,
    ):
        """
        Extend RL4CO REINFORCE loss with paper calibration loss:
        - Train calibrator (t,gamma) via BCE on (sim, correct)
        - Shape RL reward with negative BCE (log-likelihood)
        """

        # -----------------------------
        # STEP 1: get base reward
        # -----------------------------
        base_reward_raw = reward if reward is not None else policy_out["reward"]
        # keep the original reward shape for RL4CO, but use a flattened view for calibration computations
        base_reward_flat = base_reward_raw.reshape(-1)  # [B]

        # -----------------------------
        # STEP 2: normalize base_reward into a probability-friendly similarity
        # We intentionally do NOT map cosine [-1,1] -> [0,1].
        # BCE-with-logits doesn't require inputs in [0,1]; it operates on logits:
        #   logits = gamma * (sim - t)
        # so `sim` can be raw cosine-like values.
        # Use raw reward directly (no clamping) to match the equation exactly.
        sim_for_cal = base_reward_flat
        # -----------------------------

        # -----------------------------
        # STEP 3: paper BCE loss (only if td has "correct")
        # -----------------------------
        loss_cal = torch.tensor(0.0, device=sim_for_cal.device)
        r_cal = torch.zeros_like(sim_for_cal)

        if hasattr(td, "keys") and ("correct" in td.keys()):
            c = td["correct"].float().view(-1).to(sim_for_cal.device)  # [B]
            # Batch label distribution (for debugging imbalance)
            pos_cnt = None
            neg_cnt = None
            try:
                pos = (c >= 0.5).float()
                neg = 1.0 - pos
                pos_cnt = pos.sum()
                neg_cnt = neg.sum()
                denom = (pos_cnt + neg_cnt).clamp(min=1.0)
                pos_ratio = (pos_cnt / denom).detach()
                neg_ratio = (neg_cnt / denom).detach()
                # Log on step so you can see drift with different samplers (random vs anchor_nn)
                self.log_dict(
                    {
                        "train/label_pos_ratio": pos_ratio,
                        "train/label_neg_ratio": neg_ratio,
                        "train/label_pos_count": pos_cnt.detach(),
                        "train/label_neg_count": neg_cnt.detach(),
                        "train/bce_pos_weight": torch.tensor(float(getattr(self, "bce_pos_weight", 1.0)), device=sim_for_cal.device),
                        "train/bce_neg_weight": torch.tensor(float(getattr(self, "bce_neg_weight", 1.0)), device=sim_for_cal.device),
                        "train/bce_auto_balance": torch.tensor(float(bool(getattr(self, "bce_auto_balance", False))), device=sim_for_cal.device),
                    },
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )
            except Exception:
                pass

            # logits = gamma * (sim - t)
            # detach sim so BCE doesn't try to update the discrete policy
            logits = self.calibrator.logits(sim_for_cal.detach())

            # Weighted BCE to counter class imbalance.
            # We weight negatives more by setting bce_neg_weight > bce_pos_weight.
            pos_w = float(getattr(self, "bce_pos_weight", 1.0))
            neg_w = float(getattr(self, "bce_neg_weight", 1.0))

            # Optional per-batch auto-balancing weights:
            # Use inverse-frequency style weights so total contribution of pos/neg is comparable.
            auto = bool(getattr(self, "bce_auto_balance", False))
            if auto and (pos_cnt is not None) and (neg_cnt is not None):
                # If either class is missing, avoid exploding weights; fall back to 1.0
                if float(pos_cnt.detach().cpu().item()) > 0.0 and float(neg_cnt.detach().cpu().item()) > 0.0:
                    total = (pos_cnt + neg_cnt).detach()
                    # total/(2*count) gives each class ~50% total weight mass
                    auto_pos = (total / (2.0 * pos_cnt)).detach()
                    auto_neg = (total / (2.0 * neg_cnt)).detach()
                    # Combine manual and auto weights multiplicatively
                    pos_w_eff_t = auto_pos * pos_w
                    neg_w_eff_t = auto_neg * neg_w
                else:
                    pos_w_eff_t = torch.tensor(pos_w, device=sim_for_cal.device)
                    neg_w_eff_t = torch.tensor(neg_w, device=sim_for_cal.device)
            else:
                pos_w_eff_t = torch.tensor(pos_w, device=sim_for_cal.device)
                neg_w_eff_t = torch.tensor(neg_w, device=sim_for_cal.device)

            # Log effective weights (helps verify auto mode is doing what you expect)
            try:
                self.log_dict(
                    {
                        "train/bce_pos_weight_eff": pos_w_eff_t.detach(),
                        "train/bce_neg_weight_eff": neg_w_eff_t.detach(),
                    },
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )
            except Exception:
                pass

            w = torch.where(c >= 0.5, torch.full_like(c, float(pos_w_eff_t.item())), torch.full_like(c, float(neg_w_eff_t.item())))
            bce_per = F.binary_cross_entropy_with_logits(logits, c, weight=w, reduction="none")  # [B]
            loss_cal = bce_per.mean()

            # Reward shaping term (paper): -BCE, no clipping (match equation exactly).
            # Detach so this remains a scalar reward signal for REINFORCE (no direct backprop path).
            r_cal = (-bce_per).detach()

        # -----------------------------
        # STEP 4: create shaped reward for REINFORCE (paper objective)
        # Use the same similarity s that the calibrator sees (probability-friendly),
        # and add the detached log-likelihood shaping term.
        # -----------------------------
        mode = str(getattr(self, "reward_mode", "cal_only")).lower().strip()
        # signed similarity: +sim for positives, -sim for negatives
        signed_sim = sim_for_cal
        if hasattr(td, "keys") and ("correct" in td.keys()):
            c01 = td["correct"].float().view(-1).to(sim_for_cal.device)  # [B]
            sign = (2.0 * (c01 >= 0.5).float() - 1.0)  # {+1,-1}
            signed_sim = sim_for_cal * sign

        if mode == "signed_maxsim":
            shaped_reward_flat = signed_sim
        elif mode == "signed_maxsim_plus_cal":
            alpha = float(getattr(self, "alpha_cal_reward", 0.5))
            shaped_reward_flat = signed_sim + alpha * r_cal
        else:
            # cal_only: if labels are missing, fall back to base similarity
            if hasattr(td, "keys") and ("correct" in td.keys()):
                shaped_reward_flat = r_cal
            else:
                shaped_reward_flat = sim_for_cal
        shaped_reward = shaped_reward_flat.reshape_as(base_reward_raw)

        # Optional: surface key signals directly on the progress bar line.
        # Keep these short so the bar stays readable.
        try:
            self.log("reward_base", sim_for_cal.detach().mean(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("reward_cal", r_cal.detach().mean(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("reward_shaped", shaped_reward_flat.detach().mean(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            if hasattr(td, "keys") and ("correct" in td.keys()):
                c_dbg = td["correct"].float().view(-1).to(sim_for_cal.device)
                pos_ratio_dbg = (c_dbg >= 0.5).float().mean()
                self.log("pos_ratio", pos_ratio_dbg.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        except Exception:
            pass

        # -----------------------------
        # STEP 5: call RL4CO loss using shaped_reward
        # -----------------------------
        out = super().calculate_loss(
            td, batch, policy_out, reward=shaped_reward, log_likelihood=log_likelihood
        )

        # -----------------------------
        # STEP 6: add calibrator supervised loss into out["loss"]
        # so calibrator params (t,gamma) get updated.
        # -----------------------------
        beta = float(getattr(self, "beta_cal_loss", 1.0))
        if isinstance(out, dict):
            if "loss" in out and torch.is_tensor(out["loss"]):
                out["loss"] = out["loss"] + beta * loss_cal
            else:
                # fallback in case RL4CO uses a different key
                out["loss"] = beta * loss_cal

        # -----------------------------
        # STEP 7: keep your existing diagnostics (unchanged)
        # -----------------------------
        try:
            adv = out["reward"] - out["bl_val"]

            scaler = self.advantage_scaler
            adv_scaled = adv
            if getattr(scaler, "scale", None) is None:
                adv_scaled = adv
            elif isinstance(getattr(scaler, "scale", None), int):
                adv_scaled = adv / scaler.scale
            else:
                tensor_to_kwargs = dict(dtype=adv.dtype, device=adv.device)
                std = (scaler.M2 / (scaler.count - 1)).float().sqrt()
                score_scaling_factor = std.to(**tensor_to_kwargs) + torch.finfo(adv.dtype).eps
                if scaler.scale == "norm":
                    adv_scaled = (adv - scaler.mean.to(**tensor_to_kwargs)) / score_scaling_factor
                elif scaler.scale == "scale":
                    adv_scaled = adv / score_scaling_factor
                else:
                    adv_scaled = adv

            info = out.get("info", {}) if isinstance(out, dict) else {}
            ll = out["log_likelihood"].detach()
            adv_det = adv.detach()
            adv_scaled_det = adv_scaled.detach()
            dbg = {
                "train/log_likelihood_mean": ll.mean(),
                "train/log_likelihood_abs_mean": ll.abs().mean(),
                "train/advantage_mean": adv_det.mean(),
                "train/advantage_std": adv_det.std(unbiased=False),
                "train/advantage_scaled_mean": adv_scaled_det.mean(),
                "train/advantage_scaled_std": adv_scaled_det.std(unbiased=False),
                "train/baseline_mean": out["bl_val"].detach().mean(),
            }
            for k in [
                "avg_valid_a",
                "avg_valid_b",
                "min_valid_a",
                "min_valid_b",
                "frac_any_valid_a",
                "frac_any_valid_b",
                "mean_log_likelihood",
                "frac_zero_log_likelihood",
            ]:
                if isinstance(info, dict) and k in info:
                    v = info[k]
                    dbg[f"train/policy_{k}"] = v.detach() if isinstance(v, torch.Tensor) else v

            # add calibrator logs
            dbg["train/loss_cal"] = loss_cal.detach()
            dbg["train/sim_for_cal_mean"] = sim_for_cal.detach().mean()
            dbg["train/r_cal_mean"] = r_cal.detach().mean()
            dbg["train/t_hat"] = self.calibrator.t.detach()
            dbg["train/gamma_hat"] = torch.exp(self.calibrator.log_gamma.detach())
            # reward breakdown
            dbg["train/reward_base_mean"] = sim_for_cal.detach().mean()
            dbg["train/reward_cal_mean"] = r_cal.detach().mean()
            dbg["train/reward_shaped_mean"] = shaped_reward_flat.detach().mean()

            self.log_dict(dbg, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        except Exception:
            pass

        return out


class PolicyDebugCallback(Callback):
    """
    Periodically runs the current policy in debug mode on a sampled batch and appends JSONL logs.
    This is meant to help diagnose degenerate pointer selection (e.g., selecting [SEP] / no valid slots).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        log_path: str,
        every_n_epochs: int = 1,
        batch_size: int = 8,
        topk: int = 10,
        n_samples: int = 1,
        decode_type: str = "sampling",
    ):
        super().__init__()
        self.enabled = enabled
        self.log_path = log_path
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.batch_size = int(batch_size)
        self.topk = int(topk)
        self.n_samples = int(n_samples)
        self.decode_type = decode_type

    def _maybe_run(self, trainer, pl_module, tag: str):
        if not self.enabled:
            return
        # Only run on rank 0
        try:
            if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
                return
        except Exception:
            pass

        # Pull env/policy from the Lightning module
        env = getattr(pl_module, "env", None)
        policy = getattr(pl_module, "policy", None)
        if env is None or policy is None:
            return
        gen = getattr(env, "generator", None)
        if gen is None:
            return

        # Sample a batch and run debug decode
        try:
            td = gen(batch_size=self.batch_size)
            device = getattr(env, "device", None) or next(policy.parameters()).device
            td = td.to(device)
            with torch.no_grad():
                _ = policy(
                    td,
                    env,
                    phase="train",
                    decode_type=self.decode_type,
                    debug=True,
                    debug_topk=self.topk,
                    debug_n_samples=self.n_samples,
                    debug_log_path=self.log_path,
                    debug_print=True,
                )
            print(f"[POLICY_DEBUG] wrote debug_records to {self.log_path} ({tag})")
        except Exception as e:
            print(f"[POLICY_DEBUG] failed ({tag}): {e}")

    def on_train_start(self, trainer, pl_module):
        self._maybe_run(trainer, pl_module, tag="train_start")

    def on_validation_epoch_start(self, trainer, pl_module):
        # Run every N epochs
        try:
            epoch = int(getattr(trainer, "current_epoch", 0))
        except Exception:
            epoch = 0
        if epoch % self.every_n_epochs != 0:
            return
        self._maybe_run(trainer, pl_module, tag=f"val_epoch_start@epoch={epoch}")


class AnchorNNSamplerCallback(Callback):
    """Keeps the training generator in sync with the current epoch/policy for anchor-based NN sampling."""

    def __init__(self, *, enabled: bool):
        super().__init__()
        self.enabled = bool(enabled)

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.enabled:
            return
        try:
            epoch = int(getattr(trainer, "current_epoch", 0))
        except Exception:
            epoch = 0

        env = getattr(pl_module, "env", None)
        policy = getattr(pl_module, "policy", None)
        if env is None or policy is None:
            return
        gen = getattr(env, "generator", None)
        if gen is None:
            return
        if hasattr(gen, "set_epoch"):
            try:
                gen.set_epoch(epoch, policy=policy, env=env)
            except Exception as e:
                print(f"[NN_SAMPLER] generator.set_epoch failed: {e}")


def build_vocab_and_tokenize(prompts):
  
    word_to_idx = {'<pad>': 0}
    for prompt in prompts:
        tokens = prompt.lower().split()
        for token in tokens:
            if token not in word_to_idx:
                word_to_idx[token] = len(word_to_idx)
    max_len = max(len(p.lower().split()) for p in prompts)
    padded_prompts = []
    for prompt in prompts:
        tokens = prompt.lower().split()
        padding = ['<pad>'] * (max_len - len(tokens))
        ids = [word_to_idx[token] for token in tokens] + [word_to_idx[p] for p in padding]
        padded_prompts.append(ids)
    return torch.tensor(padded_prompts, dtype=torch.long), word_to_idx, max_len


def load_prompts_from_file(filepath: str) -> list:
    """从 txt 文件加载 prompts，每行一个。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            print(f"错误: 文件 {filepath} 为空或只包含空行。")
            exit(1)
        print(f"成功从 {filepath} 加载 {len(prompts)} 条 prompts。")
        return prompts
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        exit(1)
    except Exception as e:
        print(f"读取文件 {filepath} 时发生错误: {e}")
        exit(1)

def load_pairs_from_json(filepath: str) -> list:
    """Load balanced sentence pairs from JSON.

    Expected format: list[dict] with keys: sentence_1, sentence_2, correct (0/1).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        if not isinstance(pairs, list) or not pairs:
            print(f"错误: JSON 文件 {filepath} 为空或格式不是 list。")
            exit(1)
        # Light validation
        req = {"sentence_1", "sentence_2", "correct"}
        for i, ex in enumerate(pairs[:20]):
            if not isinstance(ex, dict) or not req.issubset(ex.keys()):
                print(f"错误: JSON 样本格式不正确 (index={i})，需要 keys={sorted(req)}。")
                exit(1)
        print(f"成功从 {filepath} 加载 {len(pairs)} 条 pairs。")
        return pairs
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        exit(1)
    except Exception as e:
        print(f"读取 JSON 文件 {filepath} 时发生错误: {e}")
        exit(1)


if __name__ == '__main__':
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="使用 RL4CO 训练 MaxSim 模型")
    parser.add_argument(
        '--train_file',
        type=str,
        required=False,
        default=None,
        help='包含训练 prompts 的 txt 文件路径 (例如: descriptions_train.txt)'
    )
    parser.add_argument(
        '--train_parquet',
        type=str,
        default=None,
        help="Training parquet path. Uses `--parquet_text_column` as the prompt text. Keeps other fields available via row indices.",
    )
    parser.add_argument(
        '--train_pairs_json',
        type=str,
        default=None,
        help="训练用 pairs JSON 路径（若提供则使用 sentence_1/sentence_2/correct 作为训练数据，启用 calibrator BCE）",
    )
    parser.add_argument(
        '--val_file',
        type=str,
        required=False,
        default=None,
        help='包含验证 prompts 的 txt 文件路径 (例如: descriptions_val.txt)'
    )
    parser.add_argument(
        '--val_parquet',
        type=str,
        default=None,
        help="Validation parquet path (optional).",
    )
    parser.add_argument(
        '--val_pairs_json',
        type=str,
        default=None,
        help="验证用 pairs JSON 路径（可选）",
    )
    parser.add_argument(
        '--test_file',
        type=str,
        required=False,
        default=None,
        help='包含测试 prompts 的 txt 文件路径 (例如: descriptions_test.txt)'
    )
    parser.add_argument(
        '--test_parquet',
        type=str,
        default=None,
        help="Test parquet path (optional).",
    )
    # Default to skipping test to avoid loading test parquet / prompts during training runs.
    # Use `--no-skip_test` to enable test creation + `trainer.test()` + final evaluation.
    parser.add_argument(
        "--skip_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip building the test dataset/env and skip `trainer.test()` + final test-set evaluation "
             "(default: True). Use `--no-skip_test` to run tests.",
    )
    parser.add_argument(
        "--parquet_text_column",
        type=str,
        default="prompt",
        help="Which column in the parquet contains the prompt text.",
    )
    parser.add_argument(
        "--return_row_indices",
        action="store_true",
        help="If set (and using prompts/parquet mode), generator returns row_idx_a/row_idx_b so you can look up other fields externally.",
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="none",
        choices=["none", "auto", "id_set", "string"],
        help="How to compute pair labels when training from parquet prompts: "
             "'id_set' uses ID_Set/id_set equality; 'string' compares normalized response strings; "
             "'auto' prefers id_set if usable else falls back to string only if --response_column is set; "
             "'none' disables label computation.",
    )
    parser.add_argument(
        "--response_column",
        type=str,
        default=None,
        help="Parquet column name containing the model response text to compare when --label_mode=string (e.g., response_gpt-4o-mini).",
    )
    parser.add_argument(
        '--test_pairs_json',
        type=str,
        default=None,
        help="测试用 pairs JSON 路径（可选）",
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='模型检查点（ checkpoints）的保存目录'
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="TensorBoard logger directory. Default: <checkpoint_dir>/lightning_logs (keeps logs off the root filesystem).",
    )
 
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='从指定的 checkpoint 文件路径恢复训练'
    )
    # Optional policy debug logging (writes JSONL of per-step selections)
    parser.add_argument(
        "--debug_policy",
        action="store_true",
        help="Enable periodic per-step policy selection logging (top-k, chosen token, fallback).",
    )
    parser.add_argument(
        "--debug_policy_log_path",
        type=str,
        default=None,
        help="Append JSONL policy debug records to this path (default: <checkpoint_dir>/policy_debug.jsonl).",
    )
    parser.add_argument("--debug_policy_every_n_epochs", type=int, default=1)
    parser.add_argument("--debug_policy_batch_size", type=int, default=8)
    parser.add_argument("--debug_policy_topk", type=int, default=10)
    parser.add_argument("--debug_policy_n_samples", type=int, default=1)
    parser.add_argument(
        "--save_weights_only",
        action="store_true",
        help="If set, checkpoints will store weights only (smaller, faster to save).",
    )
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="separate",
        choices=["joint", "separate"],
        help="Policy architecture mode: 'joint' uses A<->B cross-attention; 'separate' runs phi(x) and phi(y) independently (shared weights) then interleaves actions.",
    )
    parser.add_argument(
        "--train_sampling_mode",
        type=str,
        default="pairs",
        choices=["pairs", "random", "anchor_nn"],
        help="Training sampling mode. 'pairs' uses provided pairs JSON; 'random' samples random prompt pairs (requires --train_file); 'anchor_nn' uses reverse-NN anchor sampling with warmup->maxsim switch (requires --train_pairs_json).",
    )
    parser.add_argument("--nn_warmup_epochs", type=int, default=5)
    parser.add_argument("--nn_candidate_topk", type=int, default=50)
    parser.add_argument("--nn_rebuild_every_n_epochs", type=int, default=1)
    parser.add_argument(
        "--nn_similarity_mode",
        type=str,
        default="full",
        choices=["blockwise", "full"],
        help="How to compute embedding cosine NN for anchor_nn: 'blockwise' (low RAM) or 'full' (build full NxN matrix in RAM).",
    )
    parser.add_argument(
        "--nn_similarity_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Dtype for the full NxN similarity matrix when --nn_similarity_mode=full.",
    )
    parser.add_argument(
        "--nn_full_pairwise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, MaxSim NN rebuild considers all prompt pairs (O(N^2), can be very slow). Otherwise uses top-k candidates by embedding similarity.",
    )
    parser.add_argument(
        "--nn_label_strategy",
        type=str,
        default="skip",
        choices=["skip", "zero"],
        help="What to do when a sampled (x,y) pair has no label in the pairs JSON: 'skip' drops it (recommended), 'zero' sets correct=0.",
    )
    parser.add_argument(
        "--precompute_token_embeddings",
        action="store_true",
        help=(
            "If set, precompute and cache token-level embeddings (last_hidden_state, input_ids, attention_mask) "
            "for the full dataset in CPU RAM at startup. This avoids running the LM inside the generator for every batch "
            "and can greatly speed up training, at the cost of RAM."
        ),
    )
    parser.add_argument(
        "--split_on_space",
        action="store_true",
        default=False,
        help="If set, treat whitespace/word-boundary tokens as additional split delimiters (in addition to punctuation/connector words).",
    )
    parser.add_argument(
        "--split_words_before",
        action="store_true",
        default=False,
        help="If set, treat connector-word split markers as boundaries *before* the word (so the connector joins the following segment).",
    )
    parser.add_argument(
        "--punctuation_only",
        action="store_true",
        help="If set, ONLY allow punctuation tokens as valid split points (disables connector-word and whitespace splits).",
    )
    parser.add_argument(
        "--no_full_embedding",
        action="store_true",
        help="If set, do NOT append the full pooled embedding as an extra segment row during MaxSim scoring (segment-only mode).",
    )
    parser.add_argument(
        "--no_fullcos_mix",
        action="store_true",
        help=(
            "If set, do NOT mix in full-embedding cosine similarity. "
            "By default, the similarity used for reward/loss is the SIMPLE AVERAGE of: "
            "(1) MaxSim over segments (and optional full row) and (2) cosine(full_embed_a, full_embed_b)."
        ),
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use with Lightning DDP (e.g. 2,4). If >1, uses strategy='ddp'. "
             "When launched under torchrun, each process should usually use --devices=1.",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DDP find_unused_parameters. If not set, defaults to enabled when --devices > 1 "
             "(prevents DDP unused-parameter runtime error at a small performance cost).",
    )
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    parser.add_argument("--train_data_size", type=int, default=2560)
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Base RNG seed for dataset sampling. Default: -1 (non-deterministic). Set to a non-negative int for reproducible runs.",
    )
    parser.add_argument(
        "--bce_pos_weight",
        type=float,
        default=1.0,
        help="Per-example weight multiplier for BCE when correct=1.",
    )
    parser.add_argument(
        "--bce_neg_weight",
        type=float,
        default=1.0,
        help="Per-example weight multiplier for BCE when correct=0. Increase this to emphasize negatives under imbalance.",
    )
    parser.add_argument(
        "--bce_auto_balance",
        action="store_true",
        help="If set, automatically apply per-batch class-balancing weights for BCE based on the observed pos/neg counts in the batch.",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="cal_only",
        choices=["cal_only", "signed_maxsim", "signed_maxsim_plus_cal"],
        help="What scalar reward REINFORCE optimizes. "
             "'cal_only' uses reward=-BCE (loss-only reward; requires labels); "
             "'signed_maxsim' uses (+1 for correct, -1 for incorrect)*MaxSim; "
             "'signed_maxsim_plus_cal' uses signed_maxsim + alpha*(-BCE).",
    )
    args = parser.parse_args()

    # Print ALL parsed arguments in a stable order (so new flags automatically show up).
    try:
        _args_dict = vars(args)
        for _k in sorted(_args_dict.keys()):
            print(f"[ARGS] {_k}={_args_dict[_k]}")
    except Exception:
        # Fallback: don't crash training if something odd happens with argparse
        print(f"[ARGS] {args}")
    
    # Verify HF_ENDPOINT is set correctly
    hf_endpoint = os.environ.get('HF_ENDPOINT', 'Not set')
    print(f"[CONFIG] HF_ENDPOINT={hf_endpoint}")
    
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    train_pairs = load_pairs_from_json(args.train_pairs_json) if args.train_pairs_json else None
    val_pairs = load_pairs_from_json(args.val_pairs_json) if args.val_pairs_json else None
    test_pairs = load_pairs_from_json(args.test_pairs_json) if args.test_pairs_json else None

    if train_pairs is None and not args.train_file and not args.train_parquet:
        print("错误: 必须提供 --train_pairs_json 或 --train_file 或 --train_parquet")
        exit(1)
    if val_pairs is None and not args.val_file and not args.val_parquet:
        print("错误: 必须提供 --val_pairs_json 或 --val_file 或 --val_parquet")
        exit(1)
    if (not args.skip_test) and (test_pairs is None) and (not args.test_file) and (not args.test_parquet):
        print("错误: 必须提供 --test_pairs_json 或 --test_file 或 --test_parquet")
        exit(1)

    train_prompts = load_prompts_from_file(args.train_file) if (train_pairs is None and args.train_file) else None
    val_prompts = load_prompts_from_file(args.val_file) if (val_pairs is None and args.val_file) else None
    test_prompts = load_prompts_from_file(args.test_file) if (test_pairs is None and args.test_file) else None
    
  
    MAX_LEN = 512 # 文本最大长度
    MAX_SEGMENTS = 8  # 最大分割片段数
    TRAIN_DATA_SIZE_AVAILABLE = _infer_dataset_size(pairs=train_pairs, prompts=train_prompts, parquet_path=args.train_parquet)
    VAL_DATA_SIZE_AVAILABLE = _infer_dataset_size(pairs=val_pairs, prompts=val_prompts, parquet_path=args.val_parquet)
    TEST_DATA_SIZE_AVAILABLE = 0 if bool(args.skip_test) else _infer_dataset_size(pairs=test_pairs, prompts=test_prompts, parquet_path=args.test_parquet)

    # How many samples RL4CO will generate per epoch for training.
    # - In pairs mode, generator cycles through pairs (so <= available is reasonable).
    # - In random/parquet mode, generator samples with replacement (so can be > available).
    TRAIN_DATA_SIZE = int(args.train_data_size)
    VAL_DATA_SIZE = int(VAL_DATA_SIZE_AVAILABLE)
    BATCH_SIZE = int(args.batch_size)
    MAX_EPOCHS = int(args.max_epochs)  # 最大训练轮数


    print("Step 1: Instantiating components...")

    # Determine device for embedding model / env.
    # IMPORTANT: Keep this consistent with Lightning's `devices` argument below.
    # If running under DDP (Lightning spawn or torchrun), LOCAL_RANK is the per-process GPU index.
    local_rank_env = os.environ.get("LOCAL_RANK", None)
    if local_rank_env is not None:
        try:
            GPU_ID = int(local_rank_env)
        except Exception:
            GPU_ID = int(args.gpu_id)
    else:
        GPU_ID = int(args.gpu_id)
    if torch.cuda.is_available():
        n_cuda = int(torch.cuda.device_count())
        if GPU_ID < 0 or GPU_ID >= n_cuda:
            raise ValueError(
                f"Invalid --gpu_id={GPU_ID}. Visible CUDA device_count={n_cuda}. "
                "If you set CUDA_VISIBLE_DEVICES, gpu_id is relative to the visible devices (usually start at 0)."
            )
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")
    print(f"[DEVICE] Using device: {device} for embedding model / env")

    embedding_model = EmbeddingModel(device=device)

  
    base_seed = None if int(args.seed) < 0 else int(args.seed)
    train_generator = MaxSimGenerator(
        prompts=train_prompts,
        pairs=train_pairs,
        parquet_path=args.train_parquet if train_pairs is None else None,
        parquet_text_column=str(args.parquet_text_column),
        label_mode=str(args.label_mode),
        response_column=args.response_column,
        return_row_indices=bool(args.return_row_indices),
        max_len=MAX_LEN,
        embedding_model=embedding_model,
        seed=base_seed,
        sampling_mode=str(args.train_sampling_mode),
        nn_warmup_epochs=int(args.nn_warmup_epochs),
        nn_candidate_topk=int(args.nn_candidate_topk),
        nn_full_pairwise=bool(args.nn_full_pairwise),
        nn_similarity_mode=str(args.nn_similarity_mode),
        nn_similarity_dtype=str(args.nn_similarity_dtype),
        nn_rebuild_every_n_epochs=int(args.nn_rebuild_every_n_epochs),
        nn_label_strategy=str(args.nn_label_strategy),
        precompute_token_embeddings=bool(args.precompute_token_embeddings),
    )
    train_env = MaxSimEnv(
        generator=train_generator,
        max_segments=MAX_SEGMENTS,
        embedding_model=embedding_model,
        device=device,
        include_full_embedding=(not bool(args.no_full_embedding)),
        use_fullcos_mix=(not bool(args.no_fullcos_mix)),
    )

    val_generator = MaxSimGenerator(
        prompts=val_prompts,
        pairs=val_pairs,
        parquet_path=args.val_parquet if val_pairs is None else None,
        parquet_text_column=str(args.parquet_text_column),
        label_mode=str(args.label_mode),
        response_column=args.response_column,
        return_row_indices=bool(args.return_row_indices),
        sampling_mode=("pairs" if val_pairs is not None else "random"),
        max_len=MAX_LEN,
        embedding_model=embedding_model,
        seed=(None if base_seed is None else base_seed + 1),
        precompute_token_embeddings=bool(args.precompute_token_embeddings),
    )
    val_env = MaxSimEnv(
        generator=val_generator,
        max_segments=MAX_SEGMENTS,
        embedding_model=embedding_model,
        device=device,
        include_full_embedding=(not bool(args.no_full_embedding)),
        use_fullcos_mix=(not bool(args.no_fullcos_mix)),
    )
    # IMPORTANT: defer creating the test generator/env until AFTER training (or skip entirely).
    # Creating it here can load the test dataset early and cause a big memory spike at "Starting training...".
    test_env = None

  
    policy = AdaptedPointerNetworkPolicy(
        train_env,
        embedding_dim=768,
        hidden_dim=768,
        max_segments=MAX_SEGMENTS,
        policy_mode=str(args.policy_mode),
        split_on_space=(False if bool(args.punctuation_only) else bool(args.split_on_space)),
        split_words_before=(False if bool(args.punctuation_only) else bool(args.split_words_before)),
        split_on_connectors=(False if bool(args.punctuation_only) else True),
    )
    print("Components instantiated.")


    print("Step 2: Setting up RL algorithm (REINFORCE)...")

    model_kwargs = {
        'env': train_env,
        'policy': policy,
        'baseline': 'rollout',
        'train_data_size': int(args.train_data_size),
        'val_data_size': VAL_DATA_SIZE,
        'batch_size': BATCH_SIZE,
        'dataloader_num_workers': 0,
        'optimizer_kwargs': {'lr': float(args.lr)},
        'bce_pos_weight': float(args.bce_pos_weight),
        'bce_neg_weight': float(args.bce_neg_weight),
        'bce_auto_balance': bool(args.bce_auto_balance),
        'reward_mode': str(args.reward_mode),
      
    }
  
    model = ResumeFriendlyREINFORCE(**model_kwargs)
    
    model.strict_loading = False
    print("REINFORCE model configured.")


    print("Step 3: Setting up the trainer...")

    early_stopping_callback = EarlyStopping(
        monitor="val/reward",  
        mode="max",           
        patience=5,         
        verbose=True,        
        min_delta=0.01       
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='{epoch}-{step}',  
        monitor='val/reward',      
        mode='max',
        save_top_k=1,   
        save_last=True, 
        verbose=True,
        save_weights_only=bool(args.save_weights_only),
    )
    progress_bar_callback = RichProgressBar()

 
    log_dir = args.log_dir if args.log_dir else os.path.join(args.checkpoint_dir, "lightning_logs")
    logger = TensorBoardLogger(log_dir, name="maxsim_model")

    _n_devices = int(getattr(args, "devices", 1))
    _ddp_find_unused = getattr(args, "ddp_find_unused_parameters", None)
    if _ddp_find_unused is None:
        _ddp_find_unused = bool(torch.cuda.is_available() and _n_devices > 1)

    # Lightning GPU selection:
    # - If using a single GPU, pass the specific device index so we don't accidentally run on CPU
    #   or on an unintended GPU (and so our precomputed caching behavior matches the policy device).
    # - If using DDP, Lightning expects an int count.
    _trainer_devices = None
    if torch.cuda.is_available():
        if _n_devices == 1:
            _trainer_devices = [GPU_ID]
        else:
            _trainer_devices = _n_devices
    else:
        _trainer_devices = 1

    trainer = RL4COTrainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # Multi-GPU: set --devices > 1 (Lightning will launch DDP).
        # If using torchrun, pass --devices=1 and rely on LOCAL_RANK in each process.
        devices=_trainer_devices,
        # Some RL4CO baselines / policies can have conditionally unused parameters.
        # Use find_unused_parameters when requested (safer but slightly slower).
        strategy=(
            "ddp_find_unused_parameters_true"
            if (torch.cuda.is_available() and _n_devices > 1 and bool(_ddp_find_unused))
            else ("ddp" if (torch.cuda.is_available() and _n_devices > 1) else "auto")
        ),
        logger=logger, 

        callbacks=[
            early_stopping_callback,
            checkpoint_callback,
            progress_bar_callback,
            AnchorNNSamplerCallback(enabled=bool(args.train_sampling_mode == "anchor_nn")),
            PolicyDebugCallback(
                enabled=bool(args.debug_policy),
                log_path=(
                    args.debug_policy_log_path
                    if args.debug_policy_log_path
                    else os.path.join(args.checkpoint_dir, "policy_debug.jsonl")
                ),
                every_n_epochs=int(args.debug_policy_every_n_epochs),
                batch_size=int(args.debug_policy_batch_size),
                topk=int(args.debug_policy_topk),
                n_samples=int(args.debug_policy_n_samples),
                decode_type="sampling",
            ),
        ], 
        num_sanity_val_steps=1,  
        check_val_every_n_epoch=int(args.check_val_every_n_epoch),
        accumulate_grad_batches=int(args.accumulate_grad_batches),
        reload_dataloaders_every_n_epochs=1,
    )
    print("RL4COTrainer configured with Early Stopping and Checkpointing.")
    try:
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                print(f"[EarlyStopping] monitor={cb.monitor} mode={cb.mode} patience={cb.patience} min_delta={cb.min_delta}")
    except Exception as e:
        print(f"[EarlyStopping] Introspection failed: {e}")

    try:
        if hasattr(logger, "save_dir"):
            print("TB logger save_dir:", logger.save_dir)
        if hasattr(logger, "name"):
            print("TB logger name:", logger.name)
        if hasattr(logger, "version"):
            print("TB logger version:", logger.version)
    except Exception as e:
        print(f"[Logger] Introspection failed: {e}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable={trainable:,}, frozen={frozen:,}, total={trainable+frozen:,}")
    _print_cuda_mem("after_setup")

    _print_cuda_mem("before_fit")
    print("\nStarting training...")
    val_dataset = val_env.dataset(VAL_DATA_SIZE, phase="val")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=val_dataset.collate_fn)
    trainer.fit(
        model,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume_from_checkpoint
    )
    print("Training finished.")
    if bool(args.skip_test):
        print("[INFO] --skip_test set: skipping test dataset creation / trainer.test / final evaluation.")
        exit(0)

    # Build test env lazily (post-training) to avoid early memory usage.
    test_generator = MaxSimGenerator(
        prompts=test_prompts,
        pairs=test_pairs,
        parquet_path=args.test_parquet if test_pairs is None else None,
        parquet_text_column=str(args.parquet_text_column),
        label_mode=str(args.label_mode),
        response_column=args.response_column,
        return_row_indices=bool(args.return_row_indices),
        sampling_mode=("pairs" if test_pairs is not None else "random"),
        max_len=MAX_LEN,
        embedding_model=embedding_model,
        seed=(None if base_seed is None else base_seed + 2),
        precompute_token_embeddings=bool(args.precompute_token_embeddings),
    )
    test_env = MaxSimEnv(
        generator=test_generator,
        max_segments=MAX_SEGMENTS,
        embedding_model=embedding_model,
        device=device,
        include_full_embedding=(not bool(args.no_full_embedding)),
        use_fullcos_mix=(not bool(args.no_fullcos_mix)),
    )

    test_size = int(TEST_DATA_SIZE_AVAILABLE)
    test_dataset = test_env.dataset(test_size, phase="test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=test_dataset.collate_fn)
    trainer.test(model, dataloaders=test_dataloader)

    print("\n--- Step 5: Evaluating trained model on a sample from the locked-box test set ---")
    
   
   
    model.to(train_env.device) 
    model.eval()


    if test_pairs is not None and len(test_pairs) > 0:
        ex = random.choice(test_pairs)
        test_prompts_a = [ex["sentence_1"]]
        test_prompts_b = [ex["sentence_2"]]
        try:
            c = int(ex.get("correct"))
            print(f"从测试 pairs 中随机抽样进行最终评估 (correct={c}):\nA: {test_prompts_a[0]}\nB: {test_prompts_b[0]}")
        except Exception:
            print(f"从测试 pairs 中随机抽样进行最终评估:\nA: {test_prompts_a[0]}\nB: {test_prompts_b[0]}")
    elif test_prompts is not None and len(test_prompts) >= 2:
        sample_prompts = random.sample(test_prompts, 2)
        test_prompts_a = [sample_prompts[0]]
        test_prompts_b = [sample_prompts[1]]
        print(f"从测试集中随机抽样进行最终评估:\nA: {test_prompts_a[0]}\nB: {test_prompts_b[0]}")
    elif args.test_parquet:
        # Parquet mode: reuse prompts loaded by the generator (available via test_env.generator.prompts)
        try:
            prompts_list = getattr(test_env.generator, "prompts", None)
            if prompts_list is not None and len(prompts_list) >= 2:
                sample_prompts = random.sample(prompts_list, 2)
                test_prompts_a = [sample_prompts[0]]
                test_prompts_b = [sample_prompts[1]]
                print(f"从测试 parquet 中随机抽样进行最终评估:\nA: {test_prompts_a[0]}\nB: {test_prompts_b[0]}")
            else:
                raise RuntimeError("test_env.generator.prompts unavailable or too small")
        except Exception:
            print("警告: 无法从 parquet prompts 采样，使用默认 prompts 进行评估。")
            test_prompts_a = ["how to learn pytorch for deep learning"]
            test_prompts_b = ["can you give me a tutorial on pytorch tensors"]
    else:
        print("警告: 测试集样本数小于2，使用默认 prompts 进行评估。")
        test_prompts_a = ["how to learn pytorch for deep learning"]
        test_prompts_b = ["can you give me a tutorial on pytorch tensors"]
    

    # Reuse the already-built generator when possible (parquet mode needs parquet_path)
    test_generator = getattr(test_env, "generator", None)
    if test_generator is None:
        test_generator = MaxSimGenerator(
            prompts=test_prompts,
            pairs=test_pairs,
            parquet_path=args.test_parquet if test_pairs is None else None,
            parquet_text_column=str(args.parquet_text_column),
            max_len=MAX_LEN,
            embedding_model=embedding_model,
            seed=44,
        )

    test_generator.lm.to(train_env.device)
    
    inputs_a = test_generator.tokenizer(test_prompts_a, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(train_env.device)
    inputs_b = test_generator.tokenizer(test_prompts_b, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(train_env.device)
    with torch.no_grad():
        embeds_a = test_generator.lm(**inputs_a).last_hidden_state
        embeds_b = test_generator.lm(**inputs_b).last_hidden_state
    
    test_td = TensorDict({
        "token_embeddings_a": embeds_a,
        "token_embeddings_b": embeds_b,
        "attention_mask_a": inputs_a['attention_mask'],
        "attention_mask_b": inputs_b['attention_mask'],
        "length_a": inputs_a['attention_mask'].sum(dim=1),
        "length_b": inputs_b['attention_mask'].sum(dim=1),
        "input_ids_a": inputs_a['input_ids'],
        "input_ids_b": inputs_b['input_ids'],
    }, batch_size=1)

  
    N_SAMPLES = 20 
    print(f"Generating {N_SAMPLES} samples to find the best segmentation...")
    
    with torch.no_grad():
        out = model.policy(test_td.expand(N_SAMPLES), model.env, phase="test", select_best=False, decode_type="sampling")

    actions_candidates = out['actions']
    
  
    print("Evaluating each sample to find the one with the highest MaxSim score...")
    best_reward = -1.0
    best_action = None

   
    for action_candidate in actions_candidates:
        reward = train_env._get_reward(test_td, action_candidate.unsqueeze(0)).item()
        
 
        if reward > best_reward:
            best_reward = reward
            best_action = action_candidate


    print(f"\nBest segmentation found with score: {best_reward:.4f}")

  
    # Policy outputs interleaved actions: [A0, B0, A1, B1, ...]
    if best_action.numel() != 2 * MAX_SEGMENTS:
        raise ValueError(
            f"Unexpected best_action size: {best_action.numel()} (expected {2*MAX_SEGMENTS}). "
            "Check policy/env action layout."
        )
    pointers_a = best_action[0::2].tolist()  # [A0, A1, ...]
    pointers_b = best_action[1::2].tolist()  # [B0, B1, ...]

    la = int(inputs_a['attention_mask'][0].sum().item())
    lb = int(inputs_b['attention_mask'][0].sum().item())
    ids_a = inputs_a['input_ids'][0, :la]
    ids_b = inputs_b['input_ids'][0, :lb]
    tok = test_generator.tokenizer

    def _decode_segments(ids, pointers):
        length = ids.size(0)
        pointers = [min(max(0, int(p)), length - 1) for p in pointers]
        # Align with reward/policy semantics: skip [CLS] (index 0) and treat pointers as end-boundary tokens.
        bounds = sorted(set(pointers))
        segs = []
        prev = 0
        for p in bounds:
            end = p + 1
            real_start = (prev + 1) if prev > 0 else 1
            if end > real_start:
                segs.append(tok.decode(ids[real_start:end], skip_special_tokens=True).strip())
            prev = p
        tail_start = (prev + 1) if prev > 0 else 1
        if tail_start < length:
            segs.append(tok.decode(ids[tail_start:length], skip_special_tokens=True).strip())
        return segs

    segments_a = _decode_segments(ids_a, pointers_a)
    segments_b = _decode_segments(ids_b, pointers_b)

    N_PRINT = 128
    try:
        tokens_a = tok.convert_ids_to_tokens(ids_a.tolist())
    except Exception:
        tokens_a = tok.tokenize(test_prompts_a[0])[:la]
    try:
        tokens_b = tok.convert_ids_to_tokens(ids_b.tolist())
    except Exception:
        tokens_b = tok.tokenize(test_prompts_b[0])[:lb]

    def _print_tokens(label, tokens, pointers, max_n=128):
        n = min(len(tokens), max_n)
        print(f"\nTokens {label} (first {n}):")
        ptr_set = set([min(max(0, int(p)), len(tokens) - 1) for p in pointers])
        for i in range(n):
            mark = "*" if i in ptr_set else " "
            print(f"  {i:>3}{mark}: {tokens[i]}")
        ptr_info = ", ".join([f"{i}:{tokens[i]}" for i in sorted(ptr_set) if i < n])
        print(f"Pointer tokens {label}: {ptr_info}")

    print("\n--- Prompt A ---")
    print(f"Original: '{test_prompts_a[0]}'")
    _print_tokens("A", tokens_a, pointers_a, N_PRINT)
    print("Segments:")
    for i, seg in enumerate(segments_a):
        print(f"  {i+1}: '{seg}'")
        
    print("\n--- Prompt B ---")
    print(f"Original: '{test_prompts_b[0]}'")
    _print_tokens("B", tokens_b, pointers_b, N_PRINT)
    print("Segments:")
    for i, seg in enumerate(segments_b):
        print(f"  {i+1}: '{seg}'")
        
    #     python RL4COTrainer.py \
    # --train_file ../dataset/descriptions_train.txt \
    # --val_file ../dataset/descriptions_val.txt \
    # --test_file ../dataset/descriptions_test.txt \
    # --checkpoint_dir ./my_model_checkpoints
    
    #torchrun --nproc_per_node=4 RL4COTrainer.py --train_file ../dataset/descriptions_train.txt --val_file ../dataset/descriptions_val.txt --test_file ../dataset/descriptions_test.txt --checkpoint_dir ./my_model_checkpoints
    
    # python RL4COTrainer.py \
    # --train_file /path/to/descriptions_train.txt \
    # --val_file /path/to/descriptions_val.txt \
    # --test_file /path/to/descriptions_test.txt \
    # --checkpoint_dir ./my_model_checkpoints \
    # --resume_from_checkpoint ./my_model_checkpoints/last.ckpt 
