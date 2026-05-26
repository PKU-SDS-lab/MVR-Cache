import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec
)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_model import EmbeddingModel

from rl4co.envs.common.base import RL4COEnvBase


from MaxSimGenerator import MaxSimGenerator

def get_segments_from_pointers(*args, **kwargs):
    """
    Deprecated: this repo's pointers are token indices (positions in transformer input_ids),
    not word indices in `prompt.lower().split()`.

    Use `get_segments_from_token_pointers(...)` instead.
    """
    raise RuntimeError(
        "get_segments_from_pointers is deprecated because it previously assumed word-based pointers. "
        "Use get_segments_from_token_pointers(prompt, input_ids, pointers, tokenizer, ...) instead."
    )


def get_segments_from_token_pointers(
    prompt: str,
    input_ids,
    pointers: list,
    tokenizer,
    *,
    skip_cls: bool = True,
    pointers_are_end_boundaries: bool = True,
) -> list[str]:
    """
    Reconstruct string segments from token pointers in a way that matches the policy/env semantics.

    - `input_ids` is the tokenized sequence for `prompt` (1D tensor/list of ints).
    - `pointers` are token indices in that sequence.
    - When `skip_cls=True`, token index 0 is treated as [CLS] and never included in a segment.
    - When `pointers_are_end_boundaries=True`, a pointer p indicates the segment ends at token p
      (inclusive), i.e. slice end = p+1.
    """
    # Normalize input_ids to a python list[int]
    if hasattr(input_ids, "tolist"):
        ids = input_ids.tolist()
    else:
        ids = list(input_ids)
    length = len(ids)
    if length == 0:
        return [""]

    # Clamp + de-dup + sort pointers
    bounds = sorted({min(max(0, int(p)), length - 1) for p in pointers})

    segs: list[str] = []
    prev = 0
    for p in bounds:
        end = (p + 1) if pointers_are_end_boundaries else p
        start = prev
        if skip_cls:
            start = (prev + 1) if prev > 0 else 1
        if end > start:
            segs.append(tokenizer.decode(ids[start:end], skip_special_tokens=True).strip())
        prev = p

    # Tail segment
    tail_start = prev
    if skip_cls:
        tail_start = (prev + 1) if prev > 0 else 1
    if tail_start < length:
        segs.append(tokenizer.decode(ids[tail_start:length], skip_special_tokens=True).strip())

    # If everything got stripped to empty, return at least the decoded full string (sans specials)
    segs = [s for s in segs if s]
    if segs:
        return segs
    full_start = 1 if skip_cls and length > 1 else 0
    return [tokenizer.decode(ids[full_start:length], skip_special_tokens=True).strip()]
class MaxSimEnv(RL4COEnvBase):
    """
    MaxSim 环境，用于学习文本分割策略。

    该环境遵循 RL4CO 框架，定义了状态转换、奖励计算
    以及与智能体 (Policy) 的交互接口。
    """
    name = "maxsim"

    def __init__(self, 
                 generator: MaxSimGenerator, 
                 max_segments=3, 
                 lm_model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", 
                 embedding_model=None,
                 **kwargs):
        """
        初始化环境。
        """
        # Whether to append a full pooled embedding as an extra "segment" row when scoring MaxSim.
        # Default True preserves the current behavior.
        self.include_full_embedding = bool(kwargs.pop("include_full_embedding", True))
        # Whether to additionally mix in cosine(full_embed_a, full_embed_b) with MaxSim when computing
        # the similarity used for reward/loss.
        self.use_fullcos_mix = bool(kwargs.pop("use_fullcos_mix", True))
        super().__init__(check_solution=False, **kwargs) 
        
        self.generator = generator
        self.max_segments = max_segments
        
    
        if embedding_model is not None:
            self.reward_lm = embedding_model
        else:
            if "sentence-transformers" in lm_model_name_or_path or lm_model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2":
                self.reward_lm = EmbeddingModel()
            else:
                self.reward_lm = EmbeddingModel(model_name=lm_model_name_or_path)
        
        self.reward_lm.model.to(self.device)
        for param in self.reward_lm.model.parameters():
            param.requires_grad = False
        self.reward_lm.model.eval()
        
      
        # 权重顺序: [coarse, fine_row, fine_col]
        self.score_weights_raw = torch.nn.Parameter(torch.tensor([-1e9, 0.0, 0.0 ]), requires_grad=False)
        
    
        self._make_spec(self.generator)

    def dataset(self, size: int | None = None, phase: str = "train", **kwargs):
        """
        Return a **lazy** dataset that generates samples on-the-fly instead of pre-materializing
        many samples in memory.

        Compatibility:
        - RL4CO sometimes calls `env.dataset(batch_size=[N])` (baseline setup).
        - Some codepaths call `env.dataset(N, phase="val")`.
        """

        # RL4CO compatibility: allow `batch_size=[N]` as an alternative to passing `size` directly.
        if size is None:
            bs = kwargs.get("batch_size", None)
            if isinstance(bs, (list, tuple)) and len(bs) > 0:
                size = int(bs[0])
            elif isinstance(bs, torch.Size) and len(bs) > 0:
                size = int(bs[0])
            elif isinstance(bs, int):
                size = int(bs)
            else:
                raise TypeError(
                    "MaxSimEnv.dataset() requires either `size` or `batch_size=[N]`."
                )
        size = int(size)

        class _OnTheFlyTensorDictDataset(Dataset):
            def __init__(self, *, generator: MaxSimGenerator, size: int, phase: str, extra: dict | None = None):
                self.generator = generator
                self.size = int(size)
                self.phase = str(phase)
                self._extra = dict(extra or {})

                # Stable seed so __getitem__(idx) is deterministic -> required for RL4CO rollout baselines
                # that compute rewards and then call dataset.add_key("extra", rewards).
                base = getattr(generator, "seed", None)
                self._base_seed = int(base) if isinstance(base, int) else 0

            def add_key(self, key: str, value: torch.Tensor):
                """
                RL4CO baseline compatibility: return a new dataset that serves `value[idx]`
                under `key` in each sample.
                """
                return _OnTheFlyTensorDictDataset(
                    generator=self.generator,
                    size=self.size,
                    phase=self.phase,
                    extra={**self._extra, str(key): value},
                )

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # Deterministic per-index sampling so external per-index tensors (e.g., "extra")
                # align with the exact sample at that index.
                idx = int(idx)
                seed = (self._base_seed + 1_000_003 * idx) & 0xFFFFFFFF

                # Temporarily swap generator RNG for reproducible sampling.
                old_rng = getattr(self.generator, "_sample_rng", None)
                try:
                    self.generator._sample_rng = np.random.default_rng(seed)
                    td = self.generator(batch_size=1)
                finally:
                    if old_rng is not None:
                        self.generator._sample_rng = old_rng

                # Attach any extra per-index tensors (e.g., baseline rewards).
                if self._extra:
                    for k, v in self._extra.items():
                        if torch.is_tensor(v):
                            td[str(k)] = v[idx : idx + 1].to(td.device)
                        else:
                            td[str(k)] = v
                return td

            @staticmethod
            def collate_fn(batch):
                # Batch is a list[TensorDict] where each item has batch_size=[1].
                # Concatenate along batch dimension.
                return TensorDict.cat(batch, dim=0)

        return _OnTheFlyTensorDictDataset(generator=self.generator, size=size, phase=str(phase))
    def _reset(self, td: TensorDict = None, batch_size=None) -> TensorDict:
    
        if td is None:
            base_td = self.generator(batch_size=batch_size)
            td_out = base_td.empty()
            td_out.update(base_td)
        else:
            if batch_size is None:
                batch_size = td.batch_size[0]
            td_out = td.empty()
            base_td = self.generator(batch_size=batch_size)
            td_out.update(base_td)

        batch_size = td_out.batch_size[0]
        
        # 初始化
        td_out.set("current_boundary_a", torch.zeros(batch_size, dtype=torch.long, device=self.device))
        td_out.set("current_boundary_b", torch.zeros(batch_size, dtype=torch.long, device=self.device))
     
        td_out.set("i", torch.zeros(batch_size, 1, dtype=torch.long, device=self.device))#[batch_size, 1] 表示当前是第几轮 初始全0表示第一轮
        
        td_out.set("action_history", torch.zeros(batch_size, self.max_segments, 2, dtype=torch.long, device=self.device))#[batch_size, max_segments, 2] 表示每轮选择的两个边界 max_segments=3 表示每轮选择两个边界 目前有3个pointer决策

        seq_len_a = td_out["token_embeddings_a"].size(1)
        range_a = torch.arange(seq_len_a, device=self.device).expand(batch_size, -1)#batch_size行 seq_len_a列 表示每个token的索引 每行都是(0,1,2,...,seq_len_a-1)
        # Initial mask: disallow [CLS] (index 0) and exclude a trailing [SEP]/[EOS] if present.
        tok = getattr(self.reward_lm, "tokenizer", None)
        sep_id = getattr(tok, "sep_token_id", None) if tok is not None else None
        eos_id = getattr(tok, "eos_token_id", None) if tok is not None else None
        length_a = td_out["length_a"].long()
        last_idx_a = (length_a - 1).clamp(min=0)
        last_tok_a = td_out["input_ids_a"].gather(1, last_idx_a.unsqueeze(1)).squeeze(1)
        is_last_special_a = torch.zeros_like(last_tok_a, dtype=torch.bool)
        if sep_id is not None:
            is_last_special_a |= last_tok_a == sep_id
        if eos_id is not None:
            is_last_special_a |= last_tok_a == eos_id
        eff_len_a = (length_a - is_last_special_a.long()).clamp(min=2)
        # Disallow selecting the final token position (eff_len-1) as an explicit split.
        td_out.set("action_mask_a", (range_a > 0) & (range_a < (eff_len_a - 1).unsqueeze(1)))#[batch_size, seq_len_a]

        seq_len_b = td_out["token_embeddings_b"].size(1)
        range_b = torch.arange(seq_len_b, device=self.device).expand(batch_size, -1)
        length_b = td_out["length_b"].long()
        last_idx_b = (length_b - 1).clamp(min=0)
        last_tok_b = td_out["input_ids_b"].gather(1, last_idx_b.unsqueeze(1)).squeeze(1)
        is_last_special_b = torch.zeros_like(last_tok_b, dtype=torch.bool)
        if sep_id is not None:
            is_last_special_b |= last_tok_b == sep_id
        if eos_id is not None:
            is_last_special_b |= last_tok_b == eos_id
        eff_len_b = (length_b - is_last_special_b.long()).clamp(min=2)
        td_out.set("action_mask_b", (range_b > 0) & (range_b < (eff_len_b - 1).unsqueeze(1)))
        
        return td_out
    def _step(self, td: TensorDict) -> TensorDict:
        # This env was originally written with step-wise actions (shape [batch, 2]),
        # but the action_spec is a full plan (shape [batch, 2*max_segments]).
        # To avoid shape mismatches in RL4CO rollouts/baselines, support both:
        # - per-step: [batch, 2] where last dim is [b_a, b_b]
        # - full-plan: [batch, 2*max_segments] interleaved [A0,B0,A1,B1,...]
        action = td["action"]

        # Full plan case: consume entire sequence and terminate episode
        if isinstance(action, torch.Tensor) and action.dim() == 2 and action.size(1) == 2 * self.max_segments:
            plan = action.view(action.size(0), self.max_segments, 2)
            # Update history and current boundaries to the last step
            td["action_history"] = plan
            td["current_boundary_a"] = plan[:, -1, 0]
            td["current_boundary_b"] = plan[:, -1, 1]
            td["i"] = torch.full_like(td["i"], self.max_segments)

            # Masks are not used downstream in one-shot mode; keep them valid-shaped.
            done = torch.ones(action.size(0), 1, dtype=torch.bool, device=self.device)
            reward = torch.zeros_like(done, dtype=torch.float32)
            return td, reward, done

        # Per-step case
        b_a, b_b = action[:, 0], action[:, 1]

        # 更新当前边界 (Update current_boundary_A/B)
        td["current_boundary_a"] = b_a
        td["current_boundary_b"] = b_b
        
        # 将本轮动作记录到动作历史中
        td["action_history"][:, td["i"].squeeze()] = action

        #轮数加一 (Increment i)
        td["i"] = td["i"] + 1

        # Compute effective lengths that exclude trailing [SEP]/[EOS] from the selectable range.
        tok = getattr(self.reward_lm, "tokenizer", None)
        sep_id = getattr(tok, "sep_token_id", None) if tok is not None else None
        eos_id = getattr(tok, "eos_token_id", None) if tok is not None else None

        length_a = td["length_a"].long()
        length_b = td["length_b"].long()
        last_idx_a = (length_a - 1).clamp(min=0)
        last_idx_b = (length_b - 1).clamp(min=0)
        last_tok_a = td["input_ids_a"].gather(1, last_idx_a.unsqueeze(1)).squeeze(1)
        last_tok_b = td["input_ids_b"].gather(1, last_idx_b.unsqueeze(1)).squeeze(1)

        is_last_special_a = torch.zeros_like(last_tok_a, dtype=torch.bool)
        is_last_special_b = torch.zeros_like(last_tok_b, dtype=torch.bool)
        if sep_id is not None:
            is_last_special_a |= last_tok_a == sep_id
            is_last_special_b |= last_tok_b == sep_id
        if eos_id is not None:
            is_last_special_a |= last_tok_a == eos_id
            is_last_special_b |= last_tok_b == eos_id

        eff_len_a = (length_a - is_last_special_a.long()).clamp(min=2)
        eff_len_b = (length_b - is_last_special_b.long()).clamp(min=2)

        # 更新下一轮的动作掩码 (Update action_mask_A for the *next* step)
        batch_size, seq_len_a = td["action_mask_a"].shape
        range_a = torch.arange(seq_len_a, device=self.device).expand(batch_size, -1)
 
        td["action_mask_a"] = (range_a > b_a.unsqueeze(1)) & (range_a < (eff_len_a - 1).unsqueeze(1))
        
        batch_size, seq_len_b = td["action_mask_b"].shape
        range_b = torch.arange(seq_len_b, device=self.device).expand(batch_size, -1)
        td["action_mask_b"] = (range_b > b_b.unsqueeze(1)) & (range_b < (eff_len_b - 1).unsqueeze(1))

 
        # Episode is done when we either exhausted max steps OR reached the last selectable token.
        done = (td["i"] >= self.max_segments) | \
               (td["current_boundary_a"] >= (eff_len_a - 1)) | \
               (td["current_boundary_b"] >= (eff_len_b - 1))
        
        # 5. 计算奖励 (Set reward = 0.0)
        # 只有在回合结束时才计算真实奖励
        reward = torch.zeros_like(done, dtype=torch.float32)
        
        return td, reward, done

    def raw_score_text(self,
                     query_tensor: torch.Tensor,
                     sub_corpus_embeddings: torch.Tensor,
                     query_weights: torch.Tensor,
                     corpus_weights: torch.Tensor,
                     times: int = 0) -> torch.Tensor:
        """
        Compute weighted symmetric MaxSim score in raw cosine space (typically [-1, 1]).

        Notes:
        - This function is agnostic to whether a "full embedding" row is included.
          If the caller appends a full pooled vector as an extra row, it will participate
          in the MaxSim aggregation (as just another row).
        """
       
        # IMPORTANT: ensure weights are on the same device as the tensors being scored.
        # (Lightning/RL4CO can move env/policy to different CUDA devices; using self.device here
        # can cause cuda:0 vs cuda:N mismatches.)
        dev = query_tensor.device
        # Keep score_weights_raw shape unchanged for compatibility, but only use fine weights.
        fine_mix = F.softmax(self.score_weights_raw[1:].to(dev), dim=0)  # [row, col]
        w_fine_row = fine_mix[0]
        w_fine_col = fine_mix[1]

        query_vecs = query_tensor
        corpus_vecs = sub_corpus_embeddings

        # Ensure weights cover all vectors (append 1.0 weight for the full vector if needed).
        if query_weights is None:
            query_weights = torch.ones(query_vecs.shape[0], device=query_vecs.device, dtype=torch.float32)
        if corpus_weights is None:
            corpus_weights = torch.ones(corpus_vecs.shape[0], device=corpus_vecs.device, dtype=torch.float32)

        if query_weights.dim() == 1 and query_weights.shape[0] == max(0, query_vecs.shape[0] - 1):
            query_weights = torch.cat(
                [query_weights.to(dtype=torch.float32), torch.ones(1, device=query_vecs.device, dtype=torch.float32)],
                dim=0,
            )
        else:
            query_weights = query_weights.to(dtype=torch.float32)

        if corpus_weights.dim() == 1 and corpus_weights.shape[0] == max(0, corpus_vecs.shape[0] - 1):
            corpus_weights = torch.cat(
                [corpus_weights.to(dtype=torch.float32), torch.ones(1, device=corpus_vecs.device, dtype=torch.float32)],
                dim=0,
            )
        else:
            corpus_weights = corpus_weights.to(dtype=torch.float32)

        if query_vecs.shape[0] > 0 and corpus_vecs.shape[0] > 0:
            query_norm = F.normalize(query_vecs, p=2, dim=-1)
            corpus_norm = F.normalize(corpus_vecs, p=2, dim=-1)
            cos_sim_matrix = torch.mm(query_norm, corpus_norm.T)

            max_cos_sim_row = torch.max(cos_sim_matrix, dim=1).values  # [Q]
            fine_grained_row_score = torch.sum(max_cos_sim_row * query_weights) / (torch.sum(query_weights) + 1e-8)

            max_cos_sim_col = torch.max(cos_sim_matrix, dim=0).values  # [C]
            fine_grained_col_score = torch.sum(max_cos_sim_col * corpus_weights) / (torch.sum(corpus_weights) + 1e-8)
        else:
            fine_grained_row_score = torch.tensor(0.0, device=dev)
            fine_grained_col_score = torch.tensor(0.0, device=dev)

        final_score = (w_fine_row * fine_grained_row_score + w_fine_col * fine_grained_col_score)

        return final_score

    @staticmethod
    def _cos_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
        """Map cosine-like similarity from [-1, 1] to [0, 1] with clamping."""
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    def _get_reward(self, td: TensorDict, actions) -> torch.Tensor:
        batch_size = actions.shape[0]
        lm = self.reward_lm.model
        tok = self.reward_lm.tokenizer
        lm_device = next(lm.parameters()).device
        rewards = torch.zeros(batch_size, device=lm_device)

        input_ids_a = td["input_ids_a"]
        input_ids_b = td["input_ids_b"]
        length_a = td["length_a"].long()
        length_b = td["length_b"].long()

        # Compute effective lengths excluding trailing [SEP]/[EOS] from reward pooling.
        sep_id = getattr(tok, "sep_token_id", None)
        eos_id = getattr(tok, "eos_token_id", None)
        last_idx_a = (length_a - 1).clamp(min=0)
        last_idx_b = (length_b - 1).clamp(min=0)
        last_tok_a = input_ids_a.gather(1, last_idx_a.unsqueeze(1)).squeeze(1)
        last_tok_b = input_ids_b.gather(1, last_idx_b.unsqueeze(1)).squeeze(1)

        is_last_special_a = torch.zeros_like(last_tok_a, dtype=torch.bool)
        is_last_special_b = torch.zeros_like(last_tok_b, dtype=torch.bool)
        if sep_id is not None:
            is_last_special_a |= last_tok_a == sep_id
            is_last_special_b |= last_tok_b == sep_id
        if eos_id is not None:
            is_last_special_a |= last_tok_a == eos_id
            is_last_special_b |= last_tok_b == eos_id

        eff_len_a = (length_a - is_last_special_a.long()).clamp(min=2)
        eff_len_b = (length_b - is_last_special_b.long()).clamp(min=2)

        # Normalize actions -> pa_tensor/pb_tensor of shape [batch, max_segments]
        if isinstance(actions, torch.Tensor) and actions.dim() == 3 and actions.size(-1) == 2:
            # shape [batch, K, 2] where last dim is [A, B]
            pa_tensor = actions[:, :, 0]
            pb_tensor = actions[:, :, 1]
        elif isinstance(actions, torch.Tensor) and actions.dim() == 2:
            # shape [batch, 2*K] expected; policy currently outputs interleaved [A0,B0,A1,B1,...]
            if actions.size(1) != 2 * self.max_segments:
                raise ValueError(
                    f"Unsupported actions shape for reward: {actions.shape}. "
                    f"Expected second dim = {2*self.max_segments} (2*max_segments)."
                )
            pa_tensor = actions[:, 0::2]  # [A0, A1, ...]
            pb_tensor = actions[:, 1::2]  # [B0, B1, ...]
        else:
            raise ValueError(
                f"Unsupported actions shape for reward: {getattr(actions, 'shape', None)}. "
                f"Expected [batch, {2*self.max_segments}] or [batch, {self.max_segments}, 2]."
            )

        for i in range(batch_size):
            pa = pa_tensor[i].tolist()
            pb = pb_tensor[i].tolist()

            la = int(eff_len_a[i].item())
            lb = int(eff_len_b[i].item())
            la = max(1, la)
            lb = max(1, lb)

            pa = [min(max(0, p), la - 1) for p in pa]
            pb = [min(max(0, p), lb - 1) for p in pb]

            # NOTE: Align segmentation semantics with the policy.
            # Policy treats pointers as "end boundary tokens" and skips [CLS] (index 0) when building segments:
            #   real_start = prev_boundary + 1 (but starts from 1 when prev_boundary==0)
            #   real_end   = pointer + 1      (inclusive of the boundary token)
            bounds_a = sorted(set(pa))
            bounds_b = sorted(set(pb))

            # ===== ORIGINAL HEAVY LM FORWARD (保留为注释，作为兜底参考) =====
            # seg_ids_a = []
            # start = 0
            # for p in bounds_a:
            #     end = p + 1
            #     if end > start:
            #         seg_ids_a.append(input_ids_a[i, start:end])
            #     start = end
            # if start < la:
            #     seg_ids_a.append(input_ids_a[i, start:la])
            #
            # seg_ids_b = []
            # start = 0
            # for p in bounds_b:
            #     end = p + 1
            #     if end > start:
            #         seg_ids_b.append(input_ids_b[i, start:end])
            #     start = end
            # if start < lb:
            #     seg_ids_b.append(input_ids_b[i, start:lb])
            #
            # if len(seg_ids_a) == 0 or len(seg_ids_b) == 0:
            #     rewards[i] = 0.0
            #     continue
            #
            # with torch.no_grad():
            #     seg_emb_a_list = []
            #     for s in seg_ids_a:
            #         ids = s.unsqueeze(0).to(lm_device)
            #         attn = torch.ones_like(ids, device=lm_device)
            #         out = lm(ids, attention_mask=attn).last_hidden_state.mean(dim=1)
            #         seg_emb_a_list.append(out.squeeze(0))
            #     sentence_embeds_a = torch.stack(seg_emb_a_list, dim=0)
            #
            #     seg_emb_b_list = []
            #     for s in seg_ids_b:
            #         ids = s.unsqueeze(0).to(lm_device)
            #         attn = torch.ones_like(ids, device=lm_device)
            #         out = lm(ids, attention_mask=attn).last_hidden_state.mean(dim=1)
            #         seg_emb_b_list.append(out.squeeze(0))
            #     sentence_embeds_b = torch.stack(seg_emb_b_list, dim=0)
            #
            #     full_ids_a = input_ids_a[i, :la].unsqueeze(0).to(lm_device)
            #     full_attn_a = (full_ids_a != tok.pad_token_id).long()
            #     full_embed_a = lm(full_ids_a, attention_mask=full_attn_a).last_hidden_state.mean(dim=1)
            #
            #     full_ids_b = input_ids_b[i, :lb].unsqueeze(0).to(lm_device)
            #     full_attn_b = (full_ids_b != tok.pad_token_id).long()
            #     full_embed_b = lm(full_ids_b, attention_mask=full_attn_b).last_hidden_state.mean(dim=1)
            #
            #     if sentence_embeds_a.shape[0] > 0:
            #         query_tensor = torch.cat([sentence_embeds_a, full_embed_a], dim=0)
            #     else:
            #         query_tensor = full_embed_a
            #
            #     if sentence_embeds_b.shape[0] > 0:
            #         corpus_tensor = torch.cat([sentence_embeds_b, full_embed_b], dim=0)
            #     else:
            #         corpus_tensor = full_embed_b
            # ============================================================

            # FAST VERSION：直接复用 token 级嵌入做均值池化
            with torch.no_grad():
                token_emb_a = td["token_embeddings_a"][i, :la].to(lm_device)
                token_emb_b = td["token_embeddings_b"][i, :lb].to(lm_device)

                seg_emb_a_list = []
                prev = 0
                for p in bounds_a:
                    end = p + 1
                    real_start = (prev + 1) if prev > 0 else 1  # skip [CLS]
                    if end > real_start:
                        seg_emb_a_list.append(token_emb_a[real_start:end].mean(dim=0))
                    prev = p
                # Tail segment after last boundary
                tail_start = (prev + 1) if prev > 0 else 1
                if tail_start < la:
                    seg_emb_a_list.append(token_emb_a[tail_start:la].mean(dim=0))
                if len(seg_emb_a_list) == 0:
                    rewards[i] = 0.0
                    continue
                sentence_embeds_a = torch.stack(seg_emb_a_list, dim=0)

                seg_emb_b_list = []
                prev = 0
                for p in bounds_b:
                    end = p + 1
                    real_start = (prev + 1) if prev > 0 else 1  # skip [CLS]
                    if end > real_start:
                        seg_emb_b_list.append(token_emb_b[real_start:end].mean(dim=0))
                    prev = p
                tail_start = (prev + 1) if prev > 0 else 1
                if tail_start < lb:
                    seg_emb_b_list.append(token_emb_b[tail_start:lb].mean(dim=0))
                if len(seg_emb_b_list) == 0:
                    rewards[i] = 0.0
                    continue
                sentence_embeds_b = torch.stack(seg_emb_b_list, dim=0)

                # Full pooled embedding: mean over tokens excluding CLS (index 0), consistent with segment pooling.
                if int(la) > 1:
                    full_embed_a = token_emb_a[1:la].mean(dim=0, keepdim=True)
                else:
                    full_embed_a = token_emb_a.mean(dim=0, keepdim=True)
                if int(lb) > 1:
                    full_embed_b = token_emb_b[1:lb].mean(dim=0, keepdim=True)
                else:
                    full_embed_b = token_emb_b.mean(dim=0, keepdim=True)

                if self.include_full_embedding:
                    query_tensor = torch.cat([sentence_embeds_a, full_embed_a], dim=0)
                    corpus_tensor = torch.cat([sentence_embeds_b, full_embed_b], dim=0)
                else:
                    # Segment-only mode: do not append the full pooled embedding row.
                    query_tensor = sentence_embeds_a
                    corpus_tensor = sentence_embeds_b

                query_weights = torch.ones(sentence_embeds_a.shape[0], device=query_tensor.device)
                corpus_weights = torch.ones(sentence_embeds_b.shape[0], device=corpus_tensor.device)

                maxsim_score = self.raw_score_text(
                    query_tensor, corpus_tensor, query_weights, corpus_weights, times=0
                )

                if self.use_fullcos_mix:
                    # Cosine between the full pooled embeddings (single vectors).
                    # NOTE: This cosine is in raw cosine space [-1, 1], same scale as raw_score_text.
                    full_cos = F.cosine_similarity(
                        F.normalize(full_embed_a.to(query_tensor.device), p=2, dim=-1),
                        F.normalize(full_embed_b.to(query_tensor.device), p=2, dim=-1),
                        dim=-1,
                    ).mean()
                    # Simple average (no alpha): requested behavior.
                    final_score = 0.5 * (maxsim_score + full_cos)
                else:
                    final_score = maxsim_score

                # Ensure final similarity used as reward is in [0, 1].
                rewards[i] = self._cos_to_unit_interval(final_score).to(rewards.device)

        return rewards

    def _make_spec(self, generator: MaxSimGenerator):
        """
        根据生成器定义环境的 observation, action, reward, done 的规格。
        """
     
     
        sample_td = generator(batch_size=1)
        
     
        obs_spec_dict = {
            key: UnboundedContinuousTensorSpec(
                shape=val.shape,
                dtype=val.dtype,
                device=self.device
            ) for key, val in sample_td.items() if isinstance(val, torch.Tensor)
        }
        
       
        max_len = generator.max_len
        obs_spec_dict.update({
            "current_boundary_a": BoundedTensorSpec(low=0, high=max_len, shape=(1,), dtype=torch.long, device=self.device),
            "current_boundary_b": BoundedTensorSpec(low=0, high=max_len, shape=(1,), dtype=torch.long, device=self.device),
            "i": BoundedTensorSpec(low=0, high=self.max_segments, shape=(1, 1), dtype=torch.long, device=self.device),
            "action_history": BoundedTensorSpec(low=0, high=max_len, shape=(1, self.max_segments, 2), dtype=torch.long, device=self.device),
            "action_mask_a": DiscreteTensorSpec(n=2, shape=(1, max_len), dtype=torch.bool, device=self.device),
            "action_mask_b": DiscreteTensorSpec(n=2, shape=(1, max_len), dtype=torch.bool, device=self.device),
        })
        
        obs_spec = CompositeSpec(obs_spec_dict)
        
      
        # 动作是 max_segments * 2 个整数（指针）
        action_spec = BoundedTensorSpec(
            low=0,
            high=generator.max_len+1, 
            shape=(self.max_segments * 2,),
            dtype=torch.int64,
            device=self.device,
        )

       
        reward_spec = UnboundedContinuousTensorSpec(shape=(1,), device=self.device)
        
       
        done_spec = DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool, device=self.device)

        
        self.observation_spec = obs_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec
