import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec
)


from .embedding_model import EmbeddingModel

from rl4co.envs.common.base import RL4COEnvBase


from .MaxSimGenerator import MaxSimGenerator


def get_segments_from_token_pointers(
    *,
    tokenizer,
    input_ids,
    attention_mask,
    pointers: list,
) -> list[str]:
    """
    Reconstruct text segments from **token-index** pointers.

    This matches the pointer semantics used by `AdaptedPointerNetworkPolicy`:
    - pointers are 0-based token positions in `input_ids`
    - token 0 is typically [CLS] and is skipped when creating segments
    - a pointer p denotes an *inclusive* end boundary, so the slice ends at p+1
    """
    # Determine effective sequence length (exclude padding on the right).
    if attention_mask is not None:
        try:
            length = int(attention_mask.sum().item())
        except Exception:
            length = int(len(input_ids))
    else:
        length = int(len(input_ids))

    # Convert tensor -> python list of ids and clip to length.
    try:
        ids = input_ids[:length].tolist()
    except Exception:
        ids = list(input_ids)[:length]

    # Normalize pointers: unique, sorted, within range [0, length-1]
    valid_pointers = sorted({int(p) for p in pointers if 0 <= int(p) < length})

    segments: list[str] = []
    prev = 0

    # Policy semantics (see AdaptedPointerNetworkPolicy): skip [CLS] at position 0
    for p in valid_pointers:
        real_start = (prev + 1) if prev > 0 else 1
        real_end = p + 1  # inclusive boundary token
        if real_end > real_start:
            seg_text = tokenizer.decode(
                ids[real_start:real_end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if seg_text:
                segments.append(seg_text)
        prev = p

    # Tail segment after last boundary
    tail_start = (prev + 1) if prev > 0 else 1
    if tail_start < length:
        tail_text = tokenizer.decode(
            ids[tail_start:length],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if tail_text:
            segments.append(tail_text)

    # Fallback: if no segments, decode the whole (excluding [CLS])
    if not segments:
        fallback = tokenizer.decode(
            ids[1:length],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return [fallback] if fallback else [""]

    return segments


def get_segments_from_pointers(prompt: str, pointers: list) -> list:
    """
    根据原始文本和指针列表，重建文本片段。

    Args:
        prompt (str): 原始的文本字符串。
        pointers (list): 一个包含整数索引的列表，代表分割点。

    Returns:
        list: 一个包含分割后文本片段字符串的列表。
    """
    words = prompt.lower().split()
    segments = []
    last_idx = 0
    

    valid_pointers = sorted(list(set(p for p in pointers if p < len(words))))
    
    for p_idx in valid_pointers:
        # 指针 p_idx 代表在第 p_idx 个词的后面分割
        # 片段包含从 last_idx 到 p_idx (含) 的所有词
        segment = " ".join(words[last_idx : p_idx + 1])
        if segment: # 确保片段不为空
            segments.append(segment)
        last_idx = p_idx + 1
        
    # 处理最后一个片段（从最后一个指针到文本末尾）
    if last_idx < len(words):
        final_segment = " ".join(words[last_idx:])
        if final_segment:
            segments.append(final_segment)
            
 
    return segments if segments else [" ".join(words)]
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
        td_out.set("action_mask_a", range_a > 0)#[batch_size, seq_len_a] 值为1表示该位置的token可以被选择，0表示不可以被选择（pad）现在是第一位是false 其他为true 表示初始不能选index=0作为分句点

        seq_len_b = td_out["token_embeddings_b"].size(1)
        range_b = torch.arange(seq_len_b, device=self.device).expand(batch_size, -1)
        td_out.set("action_mask_b", range_b > 0)
        
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

        #更新下一轮的动作掩码 (Update action_mask_A for the *next* step)
        batch_size, seq_len_a = td["action_mask_a"].shape
        range_a = torch.arange(seq_len_a, device=self.device).expand(batch_size, -1)
 
        td["action_mask_a"] = (range_a > b_a.unsqueeze(1)) & (range_a <= td['length_a'].unsqueeze(1))
        
        batch_size, seq_len_b = td["action_mask_b"].shape
        range_b = torch.arange(seq_len_b, device=self.device).expand(batch_size, -1)
        td["action_mask_b"] = (range_b > b_b.unsqueeze(1)) & (range_b <= td['length_b'].unsqueeze(1))

 
        done = (td["i"] >= self.max_segments) | \
               (td["current_boundary_a"] >= td["length_a"]) | \
               (td["current_boundary_b"] >= td["length_b"])
        
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
        计算query和corpus之间基于粗粒度和细粒度嵌入的加权相似度分数。
        """
       
        weights = F.softmax(self.score_weights_raw, dim=0)
        w_coarse = weights[0]
        w_fine_row = weights[1]
        w_fine_col = weights[2]

        query_full_vec = query_tensor[-1:, :]
        corpus_full_vec = sub_corpus_embeddings[-1:, :]
        coarse_grained_score = F.cosine_similarity(query_full_vec, corpus_full_vec).squeeze()

        query_sentence_vecs = query_tensor[:-1, :]
        corpus_sentence_vecs = sub_corpus_embeddings[:-1, :]

        if query_sentence_vecs.shape[0] > 0 and corpus_sentence_vecs.shape[0] > 0:
            query_norm = F.normalize(query_sentence_vecs, p=2, dim=-1)
            corpus_norm = F.normalize(corpus_sentence_vecs, p=2, dim=-1)
            cos_sim_matrix = torch.mm(query_norm, corpus_norm.T)

            max_cos_sim_row = torch.max(cos_sim_matrix, dim=1).values
            fine_grained_row_score = torch.sum(max_cos_sim_row * query_weights) / (torch.sum(query_weights) + 1e-8)

            max_cos_sim_col = torch.max(cos_sim_matrix, dim=0).values
            fine_grained_col_score = torch.sum(max_cos_sim_col * corpus_weights) / (torch.sum(corpus_weights) + 1e-8)
        else:
            fine_grained_row_score = torch.tensor(0.0, device=self.device)
            fine_grained_col_score = torch.tensor(0.0, device=self.device)

      
        final_score = (w_coarse * coarse_grained_score +
                       w_fine_row * fine_grained_row_score +
                       w_fine_col * fine_grained_col_score)

        return final_score

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

            la = int(length_a[i].item())
            lb = int(length_b[i].item())
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

                full_embed_a = token_emb_a.mean(dim=0, keepdim=True)
                full_embed_b = token_emb_b.mean(dim=0, keepdim=True)

                if sentence_embeds_a.shape[0] > 0:
                    query_tensor = torch.cat([sentence_embeds_a, full_embed_a], dim=0)
                else:
                    query_tensor = full_embed_a

                if sentence_embeds_b.shape[0] > 0:
                    corpus_tensor = torch.cat([sentence_embeds_b, full_embed_b], dim=0)
                else:
                    corpus_tensor = full_embed_b

                query_weights = torch.ones(sentence_embeds_a.shape[0], device=query_tensor.device)
                corpus_weights = torch.ones(sentence_embeds_b.shape[0], device=corpus_tensor.device)

                rewards[i] = self.raw_score_text(
                    query_tensor, corpus_tensor, query_weights, corpus_weights, times=0
                ).to(rewards.device)

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