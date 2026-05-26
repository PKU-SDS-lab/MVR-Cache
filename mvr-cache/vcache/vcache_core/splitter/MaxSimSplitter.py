import importlib
import os
import re
import sys

import torch
from tensordict.tensordict import TensorDict

from .AdaptedPointerNetworkPolicy import AdaptedPointerNetworkPolicy
from .embedding_model import EmbeddingModel
from .MaxSimEnv import MaxSimEnv, get_segments_from_token_pointers
from .MaxSimGenerator import MaxSimGenerator

from rl4co.envs.common.base import RL4COEnvBase

def _register_checkpoint_module_aliases() -> None:
    """
    Backward-compat for old checkpoints saved with top-level module names.

    Example failure:
      ModuleNotFoundError: No module named 'MaxSimEnv'
    """

    aliases = {
        "MaxSimEnv": "vcache.vcache_core.splitter.MaxSimEnv",
        "MaxSimGenerator": "vcache.vcache_core.splitter.MaxSimGenerator",
        "MaxSimSplitter": "vcache.vcache_core.splitter.MaxSimSplitter",
        "embedding_model": "vcache.vcache_core.splitter.embedding_model",
        "AdaptedPointerNetworkPolicy": "vcache.vcache_core.splitter.AdaptedPointerNetworkPolicy",
    }

    for old_name, new_name in aliases.items():
        if old_name in sys.modules:
            continue
        try:
            sys.modules[old_name] = importlib.import_module(new_name)
        except Exception:
            # Best-effort; only needed if a checkpoint references this name.
            continue


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Resolve `checkpoint_path` into an actual checkpoint file on disk.

    Supported inputs:
    - A checkpoint file path (returned as-is if exists)
    - A directory path (auto-pick newest `epoch=*-step=*.ckpt`, else `last.ckpt`)
    - A non-existent file path whose parent is a directory (falls back to newest in parent)
    """
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError(f"checkpoint_path must be a non-empty str, got: {checkpoint_path!r}")

    p = os.path.abspath(os.path.expanduser(checkpoint_path))

    def _pick_from_dir(d: str) -> str:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Checkpoint directory not found: {d}")

        try:
            entries = os.listdir(d)
        except OSError as e:
            raise FileNotFoundError(f"Could not list checkpoint directory: {d} ({e})") from e

        epoch_pat = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")
        epoch_files: list[tuple[int, int, str]] = []
        for name in entries:
            m = epoch_pat.match(name)
            if not m:
                continue
            epoch_files.append((int(m.group(1)), int(m.group(2)), os.path.join(d, name)))

        if epoch_files:
            epoch_files.sort(key=lambda t: (t[0], t[1]))
            return epoch_files[-1][2]

        last_ckpt = os.path.join(d, "last.ckpt")
        if os.path.isfile(last_ckpt):
            return last_ckpt

        preview = ", ".join(sorted(entries)[:20])
        raise FileNotFoundError(
            f"No checkpoint found in directory: {d}. "
            f"Expected `epoch=*-step=*.ckpt` or `last.ckpt`. "
            f"Found (first 20): {preview}"
        )

    if os.path.isdir(p):
        return _pick_from_dir(p)

    if os.path.isfile(p):
        return p

    parent = os.path.dirname(p) or "."
    if os.path.isdir(parent):
        # Helpful fallback when a specific epoch file was referenced but rotated away.
        return _pick_from_dir(parent)

    raise FileNotFoundError(f"Checkpoint path not found: {p}")
class MaxSimSplitter:
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        embedding_model=None,
        *,
        max_segments: int = 4,
        overlap_tokens: int = 0,
        include_full_embedding: bool = False,
    ):
        # Normalize device early so all downstream `.to(...)` calls are consistent.
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.overlap_tokens = max(0, int(overlap_tokens))
        # If True, append the full pooled embedding (no-CLS mean) as an extra row to MaxSim tensors.
        # This matches the older behavior some experiments used.
        self.include_full_embedding = bool(include_full_embedding)
        resolved_ckpt = _resolve_checkpoint_path(checkpoint_path)
        print(f"正在加载分句模型: {resolved_ckpt} ...")
        
        # 1. 初始化组件
        if embedding_model is not None:
            print("MaxSimSplitter: 复用外部传入的 EmbeddingModel")
            self.embedding_model = embedding_model
        else:
            print("MaxSimSplitter: 未传入 EmbeddingModel，正在创建新实例...")
            self.embedding_model = EmbeddingModel() 

        # Ensure the underlying LM is on the requested device *before* constructing the env.
        # RL4CO env init may call the generator once (prints [GEN]) and we want it to reflect the real device.
        if hasattr(self.embedding_model, "model") and hasattr(self.embedding_model.model, "to"):
            self.embedding_model.model.to(self.device)
        
        # Dummy Generator/Env
        # NOTE: MaxSimEnv init may call the generator once (which prints `[GEN] ... device=...`).
        # Move the LM to the target device *before* constructing the env so that call uses CUDA.
        self.generator = MaxSimGenerator(
            prompts=["dummy"], max_len=512, embedding_model=self.embedding_model
        )
        if hasattr(self.generator.lm, "to"):
            self.generator.lm.to(self.device)
        # Ensure the RL4CO env itself is created on the intended device; otherwise it may move
        # the shared LM back to CPU during init.
        self.env = MaxSimEnv(
            generator=self.generator,
            max_segments=int(max_segments),
            embedding_model=self.embedding_model,
            device=self.device,
        )

        # 2. 初始化 Policy
        self.policy = AdaptedPointerNetworkPolicy(
            self.env,
            embedding_dim=768,
            hidden_dim=768,
            max_segments=int(max_segments),
        )
        
        # =================================================================
   
        original_setstate = RL4COEnvBase.__setstate__

      
        def safe_setstate(obj, state):
            try:
                original_setstate(obj, state)
            except (TypeError, RuntimeError) as e:
              
                pass

     
        RL4COEnvBase.__setstate__ = safe_setstate

        try:
            _register_checkpoint_module_aliases()
            checkpoint = torch.load(resolved_ckpt, map_location='cpu', weights_only=False)
        finally:
          
            RL4COEnvBase.__setstate__ = original_setstate
        # =================================================================

        # 4. 提取并加载权重
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            new_state_dict = {k.replace("policy.", ""): v for k, v in state_dict.items() if k.startswith("policy.")}
            # Backward-compat: older checkpoints may not contain newly-added keys
            # (e.g. env.score_weights_raw). Use strict=False but still surface unexpected
            # missing keys to avoid silently loading a wrong model.
            incompatible = self.policy.load_state_dict(new_state_dict, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if missing:
                # Allow env.* keys to be missing (env is not part of the learned splitter itself).
                non_env_missing = [k for k in missing if not k.startswith("env.")]
                if non_env_missing:
                    raise RuntimeError(
                        "Checkpoint is missing required policy weights (non-env). "
                        f"Missing keys: {non_env_missing[:20]}{' ...' if len(non_env_missing) > 20 else ''}"
                    )
                print(
                    f"Warning: checkpoint missing optional keys (ignored): "
                    f"{missing[:20]}{' ...' if len(missing) > 20 else ''}"
                )
            if unexpected:
                print(
                    f"Warning: checkpoint has unexpected keys (ignored): "
                    f"{unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}"
                )
        else:
            print("警告: Checkpoint 中没有 state_dict")
        
        # 5. 最后再移动到目标设备
        self.policy.to(self.device)
        self.policy.eval()
        print("分句模型加载完成。")
    def split_pair_return_segments(self, text_a, text_b):
        """
        输入：Query 和 Cache Candidate
        输出：RL模型优化后的 A片段列表 和 B片段列表
        """
        # 1. 构造输入 (Joint Input)
        # Batch the two texts to avoid two separate tokenizer + LM forward passes.
        inputs = self.generator.tokenizer(
            [text_a, text_b],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.inference_mode():
            hs = self.generator.lm(**inputs).last_hidden_state  # [2, L, H]

        # Split batch back into the A/B shapes expected downstream (batch_size=1 each).
        embeds_a = hs[0:1, :, :]
        embeds_b = hs[1:2, :, :]
        input_ids_a = inputs["input_ids"][0:1, :]
        input_ids_b = inputs["input_ids"][1:2, :]
        attention_mask_a = inputs["attention_mask"][0:1, :]
        attention_mask_b = inputs["attention_mask"][1:2, :]

        td = TensorDict(
            {
            "token_embeddings_a": embeds_a,
            "token_embeddings_b": embeds_b,
                "attention_mask_a": attention_mask_a,
                "attention_mask_b": attention_mask_b,
                "length_a": attention_mask_a.sum(dim=1),
                "length_b": attention_mask_b.sum(dim=1),
                "input_ids_a": input_ids_a,
                "input_ids_b": input_ids_b,
            },
            batch_size=1,
        )

        # 2. Greedy Decoding 获取最佳切分动作
        with torch.inference_mode():
            out = self.policy(
                td,
                None,
                phase="test",
                select_best=True,
                decode_type="greedy",
                compute_reward=False,
            )
        
        actions = out['actions'][0] # [2 * max_segments]

        # 3. 解析动作 -> 文本片段
        # NOTE: In this repo, actions are interleaved: [A0, B0, A1, B1, ...]
        # See `inspect_punctuation_cases.py` and `MaxSimEnv._step` (full-plan interleaved layout).
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)
        total = int(actions.numel())
        if total % 2 != 0:
            raise ValueError(f"Expected even number of action entries (A/B interleaved), got {total}")
        max_segments = total // 2
        pointers_a = actions[0: 2 * max_segments: 2].tolist()
        pointers_b = actions[1: 2 * max_segments: 2].tolist()
        
        # Reconstruct segments in **token-index space** (pointers are token positions).
        # This avoids the previous mismatch where pointers (token indices) were applied
        # to `prompt.lower().split()` (word indices).
        segments_a = get_segments_from_token_pointers(
            tokenizer=self.generator.tokenizer,
            input_ids=input_ids_a[0],
            attention_mask=attention_mask_a[0],
            pointers=pointers_a,
        )
        segments_b = get_segments_from_token_pointers(
            tokenizer=self.generator.tokenizer,
            input_ids=input_ids_b[0],
            attention_mask=attention_mask_b[0],
            pointers=pointers_b,
        )
        
        return segments_a, segments_b

    def split_pair_return_maxsim_tensors(self, text_a: str, text_b: str):
        """
        Fast path for MaxSim: run the RL splitter ONCE (tokenizer + LM forward + policy decode),
        then reuse the LM token embeddings to build:
          - query_tensor  = [segment_embeds...]
          - corpus_tensor = [segment_embeds...]

        This avoids re-encoding decoded segment strings via EmbeddingModel.get_embeddings_tensor.
        """
        import torch

        # Batch the two texts (same as split_pair_return_segments)
        inputs = self.generator.tokenizer(
            [text_a, text_b],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.inference_mode():
            hs = self.generator.lm(**inputs).last_hidden_state  # [2, L, H]

        # Split batch back into A/B (batch_size=1 each).
        token_emb_a = hs[0]  # [L, H]
        token_emb_b = hs[1]  # [L, H]
        attn_a = inputs.get("attention_mask", None)[0] if inputs.get("attention_mask", None) is not None else None
        attn_b = inputs.get("attention_mask", None)[1] if inputs.get("attention_mask", None) is not None else None
        input_ids_a = inputs["input_ids"][0:1, :]
        input_ids_b = inputs["input_ids"][1:2, :]

        td = TensorDict(
            {
                "token_embeddings_a": hs[0:1, :, :],
                "token_embeddings_b": hs[1:2, :, :],
                "attention_mask_a": inputs["attention_mask"][0:1, :] if "attention_mask" in inputs else None,
                "attention_mask_b": inputs["attention_mask"][1:2, :] if "attention_mask" in inputs else None,
                "length_a": (inputs["attention_mask"][0:1, :].sum(dim=1) if "attention_mask" in inputs else torch.tensor([hs.size(1)], device=self.device)),
                "length_b": (inputs["attention_mask"][1:2, :].sum(dim=1) if "attention_mask" in inputs else torch.tensor([hs.size(1)], device=self.device)),
                "input_ids_a": input_ids_a,
                "input_ids_b": input_ids_b,
            },
            batch_size=1,
        )

        with torch.inference_mode():
            out = self.policy(
                td,
                None,
                phase="test",
                select_best=True,
                decode_type="greedy",
                compute_reward=False,
            )

        actions = out["actions"][0]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)
        total = int(actions.numel())
        if total % 2 != 0:
            raise ValueError(f"Expected even number of action entries (A/B interleaved), got {total}")
        max_segments = total // 2
        pointers_a = actions[0 : 2 * max_segments : 2].tolist()
        pointers_b = actions[1 : 2 * max_segments : 2].tolist()

        def _length(attn, L: int) -> int:
            if attn is None:
                return int(L)
            try:
                return int(attn.sum().item())
            except Exception:
                return int(L)

        def _segment_embeds(token_emb: torch.Tensor, attn, pointers: list) -> tuple[torch.Tensor, torch.Tensor]:
            L = token_emb.shape[0]
            length = _length(attn, L)
            length = max(0, min(int(length), int(L)))
            return self._segment_embeds_from_pointers(
                token_emb, length, pointers, overlap_tokens=self.overlap_tokens
            )

        sent_a, full_a = _segment_embeds(token_emb_a, attn_a, pointers_a)
        sent_b, full_b = _segment_embeds(token_emb_b, attn_b, pointers_b)

        if self.include_full_embedding:
            query_tensor = torch.cat([sent_a, full_a], dim=0).to(dtype=torch.float32)
            corpus_tensor = torch.cat([sent_b, full_b], dim=0).to(dtype=torch.float32)
        else:
            # Default: segment embeddings only.
            query_tensor = sent_a.to(dtype=torch.float32)
            corpus_tensor = sent_b.to(dtype=torch.float32)
        return query_tensor, corpus_tensor

    def encode_text(self, text: str) -> dict:
        """
        Encode a single text once and return token-level embeddings + masks + pooled embeddings.

        Returns a dict with:
          - token_emb:      torch.Tensor [L, H] (last_hidden_state for this text)
          - attention_mask: torch.Tensor [L]
          - input_ids:      torch.Tensor [L]
          - length:         int (sum(attention_mask))
          - pooled_knn:     torch.Tensor [H]  (masked mean over *all* tokens incl. CLS, matches EmbeddingModel.get_embedding)
          - pooled_no_cls:  torch.Tensor [H]  (masked mean over tokens excluding CLS)
        """
        import torch

        inputs = self.generator.tokenizer(
            text,
            return_tensors="pt",
            padding=True,  # single example => no pad beyond itself
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.inference_mode():
            hs = self.generator.lm(**inputs).last_hidden_state  # [1, L, H]

        token_emb = hs[0]  # [L, H]
        attn = inputs.get("attention_mask", None)
        if attn is None:
            attention_mask = torch.ones((token_emb.shape[0],), device=token_emb.device, dtype=torch.long)
        else:
            attention_mask = attn[0].to(device=token_emb.device)
        input_ids = inputs["input_ids"][0]
        length = int(attention_mask.sum().item()) if attention_mask is not None else int(token_emb.shape[0])
        length = max(0, min(length, int(token_emb.shape[0])))

        # pooled_knn: masked mean over all tokens including CLS (matches EmbeddingModel.get_embedding)
        if length == 0:
            pooled_knn = token_emb.mean(dim=0)
        else:
            am = attention_mask[:length].to(dtype=token_emb.dtype).unsqueeze(-1)  # [L,1]
            pooled_knn = (token_emb[:length] * am).sum(dim=0) / am.sum(dim=0).clamp_min(1.0)

        # pooled_no_cls: masked mean excluding CLS token 0 (matches MaxSimEnv-style full embedding)
        if length <= 1:
            pooled_no_cls = pooled_knn
        else:
            am2 = attention_mask[1:length].to(dtype=token_emb.dtype).unsqueeze(-1)
            pooled_no_cls = (token_emb[1:length] * am2).sum(dim=0) / am2.sum(dim=0).clamp_min(1.0)

        return {
            "token_emb": token_emb,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "length": length,
            "pooled_knn": pooled_knn,
            "pooled_no_cls": pooled_no_cls,
        }

    @staticmethod
    def _segment_embeds_from_pointers(token_emb, length: int, pointers: list, *, overlap_tokens: int = 0):
        """
        Convert token-level embeddings + pointer indices into:
          - sentence_embeds: [S, H] (segment-level mean pooled embeddings)
          - full_embed:      [1, H] (mean pooled over all tokens excluding CLS)
        Pointer semantics:
          - pointers are token positions (0-based)
          - token 0 is [CLS] and is excluded from pooling
          - pointer p is an inclusive end boundary => slice ends at p+1
        """
        import torch
        overlap_tokens = max(0, int(overlap_tokens))

        length = max(0, min(int(length), int(token_emb.shape[0])))
        H = int(token_emb.shape[1])
        if length <= 1:
            zero = torch.zeros((1, H), device=token_emb.device, dtype=torch.float32)
            return zero, zero

        valid_pointers = sorted({int(p) for p in pointers if 0 <= int(p) < length})
        segs: list[torch.Tensor] = []
        prev = 0
        for p in valid_pointers:
            # Base semantics: start after previous boundary (skip CLS), end at p+1 (inclusive).
            base_start = (prev + 1) if prev > 0 else 1
            base_end = p + 1
            # Allow overlap: expand segment window on both sides (except never include CLS at idx 0).
            real_start = max(1, base_start - overlap_tokens)
            real_end = min(length, base_end + overlap_tokens)
            if real_end > real_start:
                segs.append(token_emb[real_start:real_end, :].mean(dim=0))
            prev = p

        base_tail_start = (prev + 1) if prev > 0 else 1
        tail_start = max(1, base_tail_start - overlap_tokens)
        if tail_start < length:
            segs.append(token_emb[tail_start:length, :].mean(dim=0))

        if not segs:
            segs = [token_emb[1:length, :].mean(dim=0)]

        sentence_embeds = torch.stack(segs, dim=0).to(dtype=torch.float32)  # [S, H]
        full_embed = token_emb[1:length, :].mean(dim=0, keepdim=True).to(dtype=torch.float32)  # [1, H]
        return sentence_embeds, full_embed

    def split_text_return_maxsim_tensor_from_encoded(self, enc: dict):
        """
        Segment a single already-encoded text using the policy's single-text decoder and
        return a MaxSim tensor:
          tensor: [S, H] (segment embeddings only; no full embedding row).
        """
        import torch

        tok = enc["token_emb"]
        am = enc["attention_mask"]
        ids = enc["input_ids"]
        length = int(enc["length"])

        # Build a single-text TensorDict in the format expected by AdaptedPointerNetworkPolicy.forward_single
        td = TensorDict(
            {
                "token_embeddings": tok.unsqueeze(0),          # [1, L, H]
                "attention_mask": am.unsqueeze(0),            # [1, L]
                "input_ids": ids.unsqueeze(0),                # [1, L]
                "length": torch.tensor([length], device=self.device),  # [1]
            },
            batch_size=1,
        )

        with torch.inference_mode():
            out = self.policy.forward_single(td, decode_type="greedy", compute_reward=False)

        actions = out["actions"][0]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)
        pointers = actions.tolist()

        sent, full = self._segment_embeds_from_pointers(
            tok, length, pointers, overlap_tokens=self.overlap_tokens
        )
        if self.include_full_embedding:
            return torch.cat([sent, full], dim=0).to(dtype=torch.float32)
        return sent.to(dtype=torch.float32)

    def split_text_return_maxsim_tensor(self, text: str):
        """
        Convenience wrapper: encode text once, then run single-text segmentation to get
        the MaxSim tensor used for similarity.
        """
        enc = self.encode_text(text)
        return self.split_text_return_maxsim_tensor_from_encoded(enc)

    def split_pair_return_maxsim_tensors_from_encoded(self, enc_a: dict, enc_b: dict) -> tuple:
        """
        Same as split_pair_return_maxsim_tensors, but reuses pre-encoded token embeddings for A and B.
        This lets the caller encode the query once (for KNN + MaxSim) and only encode the candidate once.
        """
        import torch

        # Pad to a common length for the policy.
        tok_a: torch.Tensor = enc_a["token_emb"]
        tok_b: torch.Tensor = enc_b["token_emb"]
        am_a: torch.Tensor = enc_a["attention_mask"]
        am_b: torch.Tensor = enc_b["attention_mask"]
        ids_a: torch.Tensor = enc_a["input_ids"]
        ids_b: torch.Tensor = enc_b["input_ids"]
        la: int = int(enc_a["length"])
        lb: int = int(enc_b["length"])

        L = int(max(tok_a.shape[0], tok_b.shape[0]))
        H = int(tok_a.shape[1])

        def _pad_2d(x: torch.Tensor, L: int) -> torch.Tensor:
            if x.shape[0] == L:
                return x
            pad = torch.zeros((L - x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        def _pad_1d(x: torch.Tensor, L: int) -> torch.Tensor:
            if x.shape[0] == L:
                return x
            pad = torch.zeros((L - x.shape[0],), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        tok_a_p = _pad_2d(tok_a, L)
        tok_b_p = _pad_2d(tok_b, L)
        am_a_p = _pad_1d(am_a, L)
        am_b_p = _pad_1d(am_b, L)
        ids_a_p = _pad_1d(ids_a, L)
        ids_b_p = _pad_1d(ids_b, L)

        td = TensorDict(
            {
                "token_embeddings_a": tok_a_p.unsqueeze(0),
                "token_embeddings_b": tok_b_p.unsqueeze(0),
                "attention_mask_a": am_a_p.unsqueeze(0),
                "attention_mask_b": am_b_p.unsqueeze(0),
                "length_a": torch.tensor([la], device=self.device),
                "length_b": torch.tensor([lb], device=self.device),
                "input_ids_a": ids_a_p.unsqueeze(0),
                "input_ids_b": ids_b_p.unsqueeze(0),
            },
            batch_size=1,
        )

        with torch.inference_mode():
            out = self.policy(
                td,
                None,
                phase="test",
                select_best=True,
                decode_type="greedy",
                compute_reward=False,
            )

        actions = out["actions"][0]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)
        total = int(actions.numel())
        if total % 2 != 0:
            raise ValueError(f"Expected even number of action entries (A/B interleaved), got {total}")
        max_segments = total // 2
        pointers_a = actions[0 : 2 * max_segments : 2].tolist()
        pointers_b = actions[1 : 2 * max_segments : 2].tolist()

        sent_a, full_a = self._segment_embeds_from_pointers(tok_a, la, pointers_a)
        sent_b, full_b = self._segment_embeds_from_pointers(tok_b, lb, pointers_b)

        if self.include_full_embedding:
            return (
                torch.cat([sent_a, full_a], dim=0).to(dtype=torch.float32),
                torch.cat([sent_b, full_b], dim=0).to(dtype=torch.float32),
            )

        # Default: segment embeddings only.
        return sent_a.to(dtype=torch.float32), sent_b.to(dtype=torch.float32)

    def debug_split_pair(self, text_a: str, text_b: str) -> dict:
        """
        Debug helper to inspect the splitter output.

        Returns a dict containing:
        - pointers_a / pointers_b: raw token-index pointers (inclusive end boundaries)
        - segments_a / segments_b: decoded segments reconstructed from token pointers
        - length_a / length_b: effective token lengths (sum of attention_mask)
        """
        inputs = self.generator.tokenizer(
            [text_a, text_b],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.inference_mode():
            hs = self.generator.lm(**inputs).last_hidden_state  # [2, L, H]

        embeds_a = hs[0:1, :, :]
        embeds_b = hs[1:2, :, :]
        input_ids_a = inputs["input_ids"][0:1, :]
        input_ids_b = inputs["input_ids"][1:2, :]
        attention_mask_a = inputs["attention_mask"][0:1, :]
        attention_mask_b = inputs["attention_mask"][1:2, :]

        td = TensorDict(
            {
                "token_embeddings_a": embeds_a,
                "token_embeddings_b": embeds_b,
                "attention_mask_a": attention_mask_a,
                "attention_mask_b": attention_mask_b,
                "length_a": attention_mask_a.sum(dim=1),
                "length_b": attention_mask_b.sum(dim=1),
                "input_ids_a": input_ids_a,
                "input_ids_b": input_ids_b,
            },
            batch_size=1,
        )

        with torch.inference_mode():
            out = self.policy(
                td,
                None,
                phase="test",
                select_best=True,
                decode_type="greedy",
                compute_reward=False,
                debug=True,
            )

        actions = out["actions"][0]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)
        total = int(actions.numel())
        if total % 2 != 0:
            raise ValueError(f"Expected even number of action entries (A/B interleaved), got {total}")
        max_segments = total // 2
        pointers_a = actions[0 : 2 * max_segments : 2].tolist()
        pointers_b = actions[1 : 2 * max_segments : 2].tolist()

        segments_a = get_segments_from_token_pointers(
            tokenizer=self.generator.tokenizer,
            input_ids=input_ids_a[0],
            attention_mask=attention_mask_a[0],
            pointers=pointers_a,
        )
        segments_b = get_segments_from_token_pointers(
            tokenizer=self.generator.tokenizer,
            input_ids=input_ids_b[0],
            attention_mask=attention_mask_b[0],
            pointers=pointers_b,
        )

        return {
            "pointers_a": pointers_a,
            "pointers_b": pointers_b,
            "segments_a": segments_a,
            "segments_b": segments_b,
            "length_a": int(attention_mask_a[0].sum().item()),
            "length_b": int(attention_mask_b[0].sum().item()),
            "policy_info": out.get("info", {}),
        }