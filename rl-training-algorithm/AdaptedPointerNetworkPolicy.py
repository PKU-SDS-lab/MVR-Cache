import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_model import EmbeddingModel


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.mha_ab = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.mha_ba = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.norm_ab = nn.LayerNorm(hidden_dim)
        self.norm_ba = nn.LayerNorm(hidden_dim)
        self.ff_ab = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ff_ba = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        pad_mask_a: torch.Tensor | None = None,
        pad_mask_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A attends to B
        attn_a, _ = self.mha_ab(query=emb_a, key=emb_b, value=emb_b, key_padding_mask=pad_mask_b)
        a_out = self.norm_ab(emb_a + attn_a)
        a_out = self.norm_ab(a_out + self.ff_ab(a_out))

        # B attends to A
        attn_b, _ = self.mha_ba(query=emb_b, key=emb_a, value=emb_a, key_padding_mask=pad_mask_a)
        b_out = self.norm_ba(emb_b + attn_b)
        b_out = self.norm_ba(b_out + self.ff_ba(b_out))

        return a_out, b_out

class AdaptedPointerNetworkPolicy(nn.Module):
    """
    用于双文本分割的指针网络策略。

    该策略使用交叉注意力Transformer对两个输入文本进行编码，
    并使用一个自回归的LSTM解码器来依次预测两个文本的分割边界。
    """
    def __init__(self,   
                 env, 
                 embedding_dim=768,    
                 hidden_dim=128,        
                 max_segments=6,       
                 nhead=4,               
                 num_encoder_layers=2,
                 *,
                 policy_mode: str = "joint",
                 split_on_space: bool = False,
                 split_words_before: bool = False,
                 split_on_connectors: bool = True,
                ):
        super().__init__()
        # IMPORTANT:
        # Do NOT store the full env on the policy instance.
        #
        # RL4CO's WarmupBaseline does `copy.deepcopy(policy)` during setup. Our `env` holds
        # `env.generator`, which may contain a very large CUDA token-embedding cache when
        # `--precompute_token_embeddings` is enabled (e.g., ~7.3GB for 10k x 512 x 768 fp16).
        # Keeping a strong reference here makes deepcopy try to clone those cached CUDA tensors
        # and can immediately OOM before training starts.
        #
        # RL4CO passes `env` explicitly to `forward(td, env=...)`, so retaining it is unnecessary.
        self.env = None
        self.hidden_dim = hidden_dim
        self.max_segments = max_segments
        self.policy_mode = str(policy_mode).lower().strip()
        self.split_on_space = bool(split_on_space)
        self.split_words_before = bool(split_words_before)
        self.split_on_connectors = bool(split_on_connectors)
        if self.policy_mode not in {"joint", "separate"}:
            raise ValueError(f"Unsupported policy_mode={policy_mode!r}. Use 'joint' or 'separate'.")
      
        self.train_decode_type = "sampling"
        self.val_decode_type = "greedy"
        self.test_decode_type = "greedy"
        

        self.input_proj = None

        # --- 编码器 (Encoder) ---
    
        self.encoder_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim=hidden_dim, nhead=nhead)
            for _ in range(num_encoder_layers)
        ])

        # --- 解码器 (Decoder) ---
      
        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim) 
        

        self.attention_linear_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.attention_linear_encoder = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        nn.init.uniform_(self.V, -1, 1)
        
       
        self.decoder_start_input = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        if hasattr(env, 'device'):
            self._device = env.device
        else:
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
       
        if hasattr(env, 'reward_lm'):
          
            self.embedding_model = env.reward_lm
            self.tokenizer = env.reward_lm.tokenizer
            self.lm = env.reward_lm.model
           
            self.lm.to(self._device)
            self._device = next(self.lm.parameters()).device
        elif hasattr(env, 'generator') and hasattr(env.generator, 'embedding_model'):
      
            self.embedding_model = env.generator.embedding_model
            self.tokenizer = env.generator.embedding_model.tokenizer
            self.lm = env.generator.embedding_model.model
            self.lm.to(self._device)
            self._device = next(self.lm.parameters()).device
        else:
            
            self.embedding_model = EmbeddingModel()
            self.embedding_model.model.to(self._device)
            for param in self.embedding_model.model.parameters():
                param.requires_grad = False
            self.embedding_model.model.eval()
            self.tokenizer = self.embedding_model.tokenizer
            self.lm = self.embedding_model.model
            self._device = next(self.lm.parameters()).device

        # Additional connector words that can act as split markers (in addition to punctuation).
        # We will only add those that tokenize to a single token id to avoid splitting inside subwords.
        self.split_words = [
            # Coordinating conjunctions
            "and",
            "or",
            "but",
            "nor",
            "yet",
            "so",
            # Subordinating conjunctions / complementizers
            "because",
            "although",
            "though",
            "while",
            "whereas",
            "if",
            "unless",
            "since",
            "after",
            "before",
            "when",
            "whenever",
            "once",
            "until",
            "as",
            # Relative / wh- words
            "which",
            "that",
            "who",
            "whom",
            "whose",
            "where",
            # Discourse connectives
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "nevertheless",
            "consequently",
            "meanwhile",
            "instead",
            "otherwise",
            "likewise",
            "similarly",
        ]
        if not self.split_on_connectors:
            # Punctuation-only mode: do not allow connector words to be split markers.
            self.split_words = []

        self._special_token_ids = None
        self._special_token_ids_tensor = {}
        self._word_boundary_cache: dict[int, bool] = {}
        self._word_boundary_mode: str | None = None

    @property
    def device(self) -> torch.device:
        d = getattr(self, "_device", None)
        if isinstance(d, torch.device):
            return d
        try:
            return next(self.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _init_special_token_ids(self) -> set[int]:
        if isinstance(self._special_token_ids, set):
            return self._special_token_ids
        special_ids: set[int] = set()
        tok = getattr(self, "tokenizer", None)
        for attr in [
            "pad_token_id",
            "cls_token_id",
            "sep_token_id",
            "eos_token_id",
            "bos_token_id",
            "unk_token_id",
        ]:
            v = getattr(tok, attr, None)
            if v is not None:
                try:
                    special_ids.add(int(v))
                except Exception:
                    pass
        self._special_token_ids = special_ids
        return special_ids

    def _get_special_token_ids_tensor(self, device: torch.device) -> torch.Tensor | None:
        t = self._special_token_ids_tensor.get(device)
        if isinstance(t, torch.Tensor):
            return t
        special_ids = self._init_special_token_ids()
        if not special_ids:
            self._special_token_ids_tensor[device] = None
            return None
        tt = torch.tensor(sorted(list(special_ids)), device=device, dtype=torch.long)
        self._special_token_ids_tensor[device] = tt
        return tt

    def _ensure_word_boundary_mode(self, token_strs: list[str]) -> str:
        if self._word_boundary_mode is not None:
            return self._word_boundary_mode
        mode = "wordpiece"
        for t in token_strs:
            if isinstance(t, str) and t.startswith("Ġ"):
                mode = "gpt2"
                break
            if isinstance(t, str) and t.startswith("▁"):
                mode = "sentencepiece"
                break
        self._word_boundary_mode = mode
        return mode

    def _is_word_boundary_token_str(self, t: str) -> bool:
        if not isinstance(t, str) or not t:
            return False
        mode = self._word_boundary_mode or "wordpiece"
        if mode == "gpt2":
            return t.startswith("Ġ")
        if mode == "sentencepiece":
            return t.startswith("▁")
        return not t.startswith("##")

    def _get_word_boundary_ids_tensor(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            return None
        special_ids = self._init_special_token_ids()
        uniq = torch.unique(input_ids).detach().to("cpu")
        uniq_list = [int(x) for x in uniq.tolist()]

        if self._word_boundary_mode is None:
            probe = uniq_list[: min(len(uniq_list), 4096)]
            probe_tokens = []
            for tid in probe:
                if tid in special_ids:
                    continue
                try:
                    probe_tokens.append(tok.convert_ids_to_tokens(int(tid)))
                except Exception:
                    continue
            self._ensure_word_boundary_mode(probe_tokens)

        boundary_ids: list[int] = []
        for tid in uniq_list:
            if tid in special_ids:
                continue
            cached = self._word_boundary_cache.get(tid)
            if cached is None:
                try:
                    t = tok.convert_ids_to_tokens(int(tid))
                except Exception:
                    self._word_boundary_cache[tid] = False
                    continue
                cached = self._is_word_boundary_token_str(t)
                self._word_boundary_cache[tid] = bool(cached)
            if cached:
                boundary_ids.append(tid)

        if not boundary_ids:
            return None
        return torch.tensor(boundary_ids, device=input_ids.device, dtype=torch.long)

    def _init_punctuation_ids(self, device):
        """
        初始化有效的分割token ID集合。
        包含: 基础标点 + 带空格的标点变体 + 可选连接词 split_words。

        IMPORTANT: Do NOT include special tokens like [SEP]/[EOS]/[CLS]/[PAD] as "punctuation".
        If [SEP] is treated as a valid split point, the policy can end immediately by choosing it
        on step 0, then the "future boundary" constraint makes all subsequent steps invalid,
        leading to repeated fallbacks and degenerate behavior.
        """
      
        if hasattr(self, "_valid_split_ids") and self._valid_split_ids is not None:
            if self._valid_split_ids.device == device:
                return
        
      
        punct_chars = {",", ".", "!", "?", ":", ";", "，", "。", "！", "？", "：", "；"}
        punct_ids = set()
        connector_ids = set()
            
  
        for char in punct_chars:
            # Case A: 纯标点
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            if ids: punct_ids.update(ids)
            
            # Case B: 带空格前缀
            ids_space = self.tokenizer.encode(" " + char, add_special_tokens=False)
            if ids_space: punct_ids.update(ids_space)

        # Add connector words as split markers (single-token only).
        for w in getattr(self, "split_words", []) or []:
            # Include both raw and space-prefixed forms for different tokenizer families.
            for ww in {w, w.capitalize()}:
                for prefix in ("", " "):
                    ids = self.tokenizer.encode(prefix + ww, add_special_tokens=False)
                    # Only accept markers that map to a single token id to avoid mid-word splits.
                    if isinstance(ids, list) and len(ids) == 1:
                        connector_ids.add(ids[0])

     
        sorted_punct_ids = sorted(list(punct_ids))
        sorted_connector_ids = sorted(list(connector_ids))

        self._punct_split_ids = (
            torch.tensor(sorted_punct_ids, device=device, dtype=torch.long)
            if sorted_punct_ids
            else torch.empty((0,), device=device, dtype=torch.long)
        )
        self._connector_split_ids = (
            torch.tensor(sorted_connector_ids, device=device, dtype=torch.long)
            if sorted_connector_ids
            else torch.empty((0,), device=device, dtype=torch.long)
        )

        sorted_ids = sorted(list(punct_ids | connector_ids))
        self._valid_split_ids = (
            torch.tensor(sorted_ids, device=device, dtype=torch.long)
            if sorted_ids
            else torch.empty((0,), device=device, dtype=torch.long)
        )

    def _compute_split_candidate_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_punct_global = (
            torch.isin(input_ids, self._punct_split_ids)
            if hasattr(self, "_punct_split_ids") and isinstance(self._punct_split_ids, torch.Tensor)
            else torch.zeros_like(input_ids, dtype=torch.bool)
        )

        if hasattr(self, "_connector_split_ids") and isinstance(
            self._connector_split_ids, torch.Tensor
        ):
            if int(self._connector_split_ids.numel()) > 0:
                is_conn = torch.isin(input_ids, self._connector_split_ids)
                if self.split_words_before:
                    is_conn_before = torch.zeros_like(is_conn)
                    is_conn_before[:, :-1] = is_conn[:, 1:]
                    is_conn_before[:, 0] = False
                    sp_t = self._get_special_token_ids_tensor(input_ids.device)
                    if isinstance(sp_t, torch.Tensor) and int(sp_t.numel()) > 0:
                        is_conn_before = is_conn_before & (~torch.isin(input_ids, sp_t))
                    is_punct_global = is_punct_global | is_conn_before
                else:
                    is_punct_global = is_punct_global | is_conn

        if self.split_on_space:
            word_ids = self._get_word_boundary_ids_tensor(input_ids)
            if word_ids is not None:
                is_punct_global = is_punct_global | torch.isin(input_ids, word_ids)

        return is_punct_global

    def _decode_single(
        self,
        *,
        encoder_outputs: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        eff_len: torch.Tensor,  # [B]
        decode_type: str,
        debug: bool = False,
        debug_topk: int = 8,
        debug_n_samples: int = 1,
        side: str = "S",
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Decode segmentation pointers for a single sequence (factorized policy mode).

        Returns:
        - pointers: [B, max_segments] (token positions)
        - logp: [B] (sum log-prob over steps)
        - info: dict of diagnostics + optional debug_records
        """
        batch_size, seq_len, _ = encoder_outputs.shape

        # Precompute split-marker mask [B, L]
        is_punct_global = self._compute_split_candidate_mask(input_ids)

        # Decoder init
        dev = encoder_outputs.device
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1).to(dev)
        h = torch.zeros(batch_size, self.decoder_cell.hidden_size, device=dev)
        c = torch.zeros(batch_size, self.decoder_cell.hidden_size, device=dev)
        current_b = torch.zeros(batch_size, dtype=torch.long, device=dev)

        pointers_steps = []
        logp_steps = []
        valid_counts_steps = []
        had_any_valid_steps = []
        debug_records = []

        # Precompute ctx [B, H, L] for efficiency inside the loop
        ctx = self.attention_linear_encoder(encoder_outputs.permute(0, 2, 1))
        V_exp = self.V.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).to(dev)
        mask_fill_val = float("-inf")

        for step in range(self.max_segments):
            h, c = self.decoder_cell(decoder_input, (h, c))
            query_vec = self.attention_linear_decoder(h)

            rng = torch.arange(seq_len, device=dev).expand(batch_size, -1)
            is_future = rng > current_b.unsqueeze(1)
            is_in_len = rng < (eff_len - 1).unsqueeze(1)
            valid_slots = is_punct_global & is_future & is_in_len
            mask_ptr = ~valid_slots

            valid_counts_steps.append(valid_slots.sum(dim=1))
            had_any_valid_steps.append(valid_slots.any(dim=1))

            inp = query_vec.unsqueeze(2)  # [B, H, 1]
            attn_scores = torch.bmm(V_exp, torch.tanh(inp + ctx)).squeeze(1)  # [B, L]
            attn_scores = attn_scores.masked_fill(mask_ptr, mask_fill_val)

            pointer = torch.zeros(batch_size, dtype=torch.long, device=dev)
            logp = torch.zeros(batch_size, device=dev)

            has_valid = valid_slots.any(dim=1)
            if has_valid.any():
                rows = has_valid
                if decode_type == "sampling":
                    dist = Categorical(logits=attn_scores[rows])
                    act = dist.sample()
                    pointer[rows] = act
                    logp[rows] = dist.log_prob(act)
                else:
                    pointer[rows] = torch.argmax(attn_scores[rows], dim=1)

            if (~has_valid).any():
                rows = ~has_valid
                pointer[rows] = (eff_len[rows] - 1).clamp(min=1)

            if debug:
                # Capture step-level debug info for the first N samples
                n_dbg = min(int(debug_n_samples), batch_size)
                for bi in range(n_dbg):
                    pb = int(pointer[bi].item())
                    ids_row = input_ids[bi]

                    def _pos_to_tok(seq_ids, pos):
                        try:
                            tok_id = int(seq_ids[pos].item())
                        except Exception:
                            tok_id = int(seq_ids[pos])
                        try:
                            return self.tokenizer.convert_ids_to_tokens(tok_id)
                        except Exception:
                            return str(tok_id)

                    debug_records.append(
                        {
                            "side": side,
                            "step": int(step),
                            "sample": int(bi),
                            "current_boundary": int(current_b[bi].item()),
                            "eff_len": int(eff_len[bi].item()),
                            "n_valid": int(valid_slots[bi].sum().item()),
                            "chosen_pos": pb,
                            "chosen_tok": _pos_to_tok(ids_row, pb),
                            "fallback": bool((~has_valid)[bi].item()),
                            "chosen_is_valid": bool(valid_slots[bi, pb].item()) if pb < valid_slots.size(1) else False,
                        }
                    )

            # Segment pooling for feedback (uses encoder_outputs, already in hidden_dim)
            feedback = torch.zeros(batch_size, self.hidden_dim, device=dev)
            for b in range(batch_size):
                s = int(current_b[b].item())
                e = int(pointer[b].item())
                real_start = (s + 1) if s > 0 else 1  # skip [CLS]
                real_end = e + 1  # inclusive boundary
                if real_end > real_start:
                    feedback[b] = encoder_outputs[b, real_start:real_end].mean(dim=0)
                else:
                    feedback[b].zero_()
            decoder_input = feedback

            current_b = pointer
            pointers_steps.append(pointer)
            logp_steps.append(logp)

        pointers = torch.stack(pointers_steps, dim=1)  # [B, K]
        logp_total = torch.stack(logp_steps, dim=1).sum(dim=1)  # [B]

        info = {}
        try:
            vc = torch.stack(valid_counts_steps, dim=1).float()
            hv = torch.stack(had_any_valid_steps, dim=1)
            info.update(
                {
                    f"avg_valid_{side.lower()}": vc.mean().detach(),
                    f"min_valid_{side.lower()}": vc.min().detach(),
                    f"frac_any_valid_{side.lower()}": hv.float().mean().detach(),
                }
            )
        except Exception:
            pass
        if debug:
            info["debug_records"] = debug_records

        return pointers, logp_total, info

    def forward(self, td, env=None, phase: str = "train", select_best: bool = False, **kwargs):
        """与 RL4CO 的 REINFORCE 调用约定兼容的入口。
        根据阶段/标志选择采样或贪心解码，返回包含 reward 与 log_likelihood 的字典。
        """
       
        decode_override = kwargs.pop("decode_type", None)
        if decode_override is not None:
            decode_type = decode_override
        else:
            decode_type = "greedy" if (select_best or phase != "train") else "sampling"

        actions, log_likelihood, info = self._forward(td, decode_type=decode_type, **kwargs)
        # 计算奖励
        # We intentionally do not fall back to a stored env (see __init__ note). Callers must pass it.
        used_env = env
        if used_env is None:
            raise RuntimeError("Environment instance is required; call forward(..., env=your_env).")
        reward = used_env.get_reward(td, actions)

        return {
            "actions": actions,
            "log_likelihood": log_likelihood,
            "reward": reward,
            "info": info,
        }

    def _move_td_keys_to_device(self, td, keys: list[str], device: torch.device):
        """Best-effort helper: move a subset of TensorDict keys to device."""
        try:
            return td.to(device)
        except Exception:
            for k in keys:
                if k in td and isinstance(td[k], torch.Tensor):
                    td[k] = td[k].to(device)
            return td

    def forward_single(
        self,
        td,
        *,
        decode_type: str = "sampling",
        key_suffix: str | None = None,
        side: str = "S",
        **kwargs,
    ):
        """
        Single-text policy forward φ(text): runs the pointer decoder on ONE sequence only.

        This is the API you asked for: you can call it twice as:
          out_a = policy.forward_single(td_pair, key_suffix="a", side="A")
          out_b = policy.forward_single(td_pair, key_suffix="b", side="B")
        then interleave actions and compute pair reward.

        Supported input formats:
        - Pair TensorDict (current generator/env format): keys end with `_a` / `_b`
          (use key_suffix="a" or "b")
        - Single TensorDict: keys without suffixes: token_embeddings, attention_mask, input_ids, length
          (use key_suffix=None)

        Returns dict with:
        - actions: [B, max_segments]  (pointer positions)
        - log_likelihood: [B]
        - info: dict (may include debug_records)

        NOTE: This does NOT compute reward (your reward is pair-based in MaxSimEnv).
        """
        debug = bool(kwargs.pop("debug", False))
        debug_topk = int(kwargs.pop("debug_topk", 8))
        debug_n_samples = int(kwargs.pop("debug_n_samples", 1))

        # Ensure module + data are on the LM/policy device
        target_device = next(self.lm.parameters()).device
        if getattr(self, "_current_module_device", None) != target_device:
            self.to(target_device)
            self._current_module_device = target_device
        self._device = target_device

        if key_suffix is None:
            emb_k = "token_embeddings"
            attn_k = "attention_mask"
            ids_k = "input_ids"
            len_k = "length"
        else:
            suf = str(key_suffix).lower().strip()
            if suf not in {"a", "b"}:
                raise ValueError(f"key_suffix must be 'a', 'b', or None; got {key_suffix!r}")
            emb_k = f"token_embeddings_{suf}"
            attn_k = f"attention_mask_{suf}"
            ids_k = f"input_ids_{suf}"
            len_k = f"length_{suf}"

        td = self._move_td_keys_to_device(td, [emb_k, attn_k, ids_k, len_k], target_device)

        # Initialize split markers (punctuation/connector ids) once per device
        self._init_punctuation_ids(td[ids_k].device)

        # Lazily build projection to hidden_dim
        input_dim = td[emb_k].size(-1)
        if self.input_proj is None:
            with torch.inference_mode(False):
                if input_dim == self.hidden_dim:
                    self.input_proj = nn.Identity()
                else:
                    self.input_proj = nn.Linear(input_dim, self.hidden_dim)
            self.input_proj.to(self.device)

        encoder_outputs = self.input_proj(td[emb_k])  # [B, L, H]

        # Effective length excludes trailing [SEP]/[EOS] from selectable range
        length = td[len_k].long()
        last_idx = (length - 1).clamp(min=0)
        last_tok = td[ids_k].gather(1, last_idx.unsqueeze(1)).squeeze(1)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        is_last_special = torch.zeros_like(last_tok, dtype=torch.bool)
        if sep_id is not None:
            is_last_special |= last_tok == sep_id
        if eos_id is not None:
            is_last_special |= last_tok == eos_id
        eff_len = (length - is_last_special.long()).clamp(min=2)

        pointers, logp, info = self._decode_single(
            encoder_outputs=encoder_outputs,
            input_ids=td[ids_k],
            eff_len=eff_len,
            decode_type=str(decode_type),
            debug=debug,
            debug_topk=debug_topk,
            debug_n_samples=debug_n_samples,
            side=str(side),
        )

        return {"actions": pointers, "log_likelihood": logp, "info": info}

    def _forward(self, td, decode_type="sampling", **kwargs):
        """
        实现自回归解码的核心逻辑 。
        """
        # Optional per-step debug logging (intended for inspection scripts, not training).
        debug = bool(kwargs.pop("debug", False))
        debug_topk = int(kwargs.pop("debug_topk", 8))
        debug_n_samples = int(kwargs.pop("debug_n_samples", 1))
        debug_log_path = kwargs.pop("debug_log_path", None)
        debug_print = bool(kwargs.pop("debug_print", True))
        debug_records = []
       
     
        target_device = next(self.lm.parameters()).device
        if getattr(self, "_current_module_device", None) != target_device:
            self.to(target_device)
            self._current_module_device = target_device
        self._device = target_device
        
   
        try:
            td = td.to(target_device)
        except Exception:
            for k in ['token_embeddings_a','token_embeddings_b','attention_mask_a','attention_mask_b',
                      'input_ids_a','input_ids_b','length_a','length_b']:
                if k in td and isinstance(td[k], torch.Tensor):
                    td[k] = td[k].to(target_device)
        self._init_punctuation_ids(td['input_ids_a'].device)
        
     
        input_dim = td['token_embeddings_a'].size(-1)
        if self.input_proj is None:
            with torch.inference_mode(False):
                if input_dim == self.hidden_dim:
                    self.input_proj = nn.Identity()
                else:
                    self.input_proj = nn.Linear(input_dim, self.hidden_dim)
            self.input_proj.to(self.device)

        embedded_a = self.input_proj(td['token_embeddings_a'])
        embedded_b = self.input_proj(td['token_embeddings_b'])
        
        mask_a = ~td['attention_mask_a'].bool()
        mask_b = ~td['attention_mask_b'].bool()
        
        batch_size = embedded_a.size(0)
        seq_len_a = embedded_a.size(1)
        seq_len_b = embedded_b.size(1)

        # Compute "effective lengths" that exclude trailing [SEP]/[EOS] from the selectable range.
        length_a = td["length_a"].long()
        length_b = td["length_b"].long()
        last_idx_a = (length_a - 1).clamp(min=0)
        last_idx_b = (length_b - 1).clamp(min=0)
        last_tok_a = td["input_ids_a"].gather(1, last_idx_a.unsqueeze(1)).squeeze(1)
        last_tok_b = td["input_ids_b"].gather(1, last_idx_b.unsqueeze(1)).squeeze(1)

        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

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

        # ============================
        # Separate (factorized) mode:
        # run φ(x) and φ(y) independently (shared weights), then interleave actions.
        # ============================
        if self.policy_mode == "separate":
            # Use the single-text API twice (φ(x), φ(y)), then combine.
            out_a = self.forward_single(
                td,
                decode_type=decode_type,
                key_suffix="a",
                side="A",
                debug=debug,
                debug_topk=debug_topk,
                debug_n_samples=debug_n_samples,
            )
            out_b = self.forward_single(
                td,
                decode_type=decode_type,
                key_suffix="b",
                side="B",
                debug=debug,
                debug_topk=debug_topk,
                debug_n_samples=debug_n_samples,
            )
            pa, logp_a, info_a = out_a["actions"], out_a["log_likelihood"], out_a.get("info", {})
            pb, logp_b, info_b = out_b["actions"], out_b["log_likelihood"], out_b.get("info", {})

            plan = torch.stack([pa, pb], dim=2)  # [B, K, 2] => [A0,B0,...]
            actions = plan.view(batch_size, -1)  # [B, 2*K]
            log_likelihood = logp_a + logp_b

            info = {}
            info.update({k: v for k, v in info_a.items() if k != "debug_records"})
            info.update({k: v for k, v in info_b.items() if k != "debug_records"})
            if debug:
                # Merge debug records if present
                recs = []
                recs.extend(info_a.get("debug_records", []))
                recs.extend(info_b.get("debug_records", []))
                info["debug_records"] = recs

                if debug_print:
                    for r in recs:
                        if "chosen_pos" in r:
                            print(
                                f"[DBG][{r.get('side','?')}] step={r['step']} sample={r['sample']} "
                                f"pos={r['chosen_pos']} tok={r.get('chosen_tok')} "
                                f"valid={r.get('chosen_is_valid')} fallback={r.get('fallback')}"
                            )
                if debug_log_path:
                    try:
                        import json
                        with open(debug_log_path, "a", encoding="utf-8") as f:
                            for r in recs:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    except Exception as e:
                        info["debug_log_error"] = str(e)

            return actions, log_likelihood, info
        
        # --- 2. Encoder (Cross-Attention) ---
        for layer in self.encoder_layers:
            embedded_a, embedded_b = layer(
                emb_a=embedded_a, 
                emb_b=embedded_b, 
                pad_mask_a=mask_a, 
                pad_mask_b=mask_b
            )
        encoder_outputs_a = embedded_a
        encoder_outputs_b = embedded_b
        
        # --- 3. Decoder 初始化 ---
        dev = self.device
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1).to(dev)
        h, c = (torch.zeros(batch_size, self.decoder_cell.hidden_size, device=dev),
                torch.zeros(batch_size, self.decoder_cell.hidden_size, device=dev))
        
        pointers = []
        log_probs = []
        # Debug stats to diagnose "loss not improving" (e.g., degenerate action space => logp ~ 0)
        valid_counts_a_steps = []
        valid_counts_b_steps = []
        had_any_valid_a_steps = []
        had_any_valid_b_steps = []
        
        # 初始边界设为 0 (对应 [CLS])
        current_bA = torch.zeros(batch_size, dtype=torch.long, device=dev)
        current_bB = torch.zeros(batch_size, dtype=torch.long, device=dev)

        # 预计算全局标点 Mask [batch, seq_len]
    
        is_punct_global_a = self._compute_split_candidate_mask(td['input_ids_a'])
        is_punct_global_b = self._compute_split_candidate_mask(td['input_ids_b'])

        # Ensure we always keep at least [CLS] plus one token in range computations
        eff_len_a = eff_len_a.clamp(min=2)
        eff_len_b = eff_len_b.clamp(min=2)

        # --- 4. 解码循环 ---
        for step in range(self.max_segments):
            # (a) LSTM Step
            h, c = self.decoder_cell(decoder_input, (h, c))
            query_vec = self.attention_linear_decoder(h)

            #  指针 A 选择
            range_a = torch.arange(seq_len_a, device=dev).expand(batch_size, -1)
            
            # 1. 必须是标点符号 (预计算结果)
            # 2. 必须严格在当前边界之后 (range_a > current_bA)
            # 3. 必须在有效长度内 (range_a < length_a)
       
            
            is_future_a = range_a > current_bA.unsqueeze(1)
            # Disallow selecting the final token position (eff_len-1) as an explicit split,
            # since that creates a single "whole text" segment. End-of-text should be reached via fallback.
            is_in_length_a = range_a < (eff_len_a - 1).unsqueeze(1)
            valid_slots_a = is_punct_global_a & is_future_a & is_in_length_a
            mask_ptr_a = ~valid_slots_a
            valid_counts_a_steps.append(valid_slots_a.sum(dim=1))
            had_any_valid_a_steps.append(valid_slots_a.any(dim=1))

            inp_a = query_vec.unsqueeze(2)
            ctx_a = self.attention_linear_encoder(encoder_outputs_a.permute(0, 2, 1))
            V_exp = self.V.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).to(dev)
            attn_scores_a = torch.bmm(V_exp, torch.tanh(inp_a + ctx_a)).squeeze(1)
            
            # ...
            mask_fill_val = float('-inf') 
            attn_scores_a = attn_scores_a.masked_fill(mask_ptr_a, mask_fill_val)

            # === 指针 B 选择 ===
            range_b = torch.arange(seq_len_b, device=dev).expand(batch_size, -1)
            
            is_future_b = range_b > current_bB.unsqueeze(1)
            is_in_length_b = range_b < (eff_len_b - 1).unsqueeze(1)
            
            valid_slots_b = is_punct_global_b & is_future_b & is_in_length_b
            mask_ptr_b = ~valid_slots_b
            valid_counts_b_steps.append(valid_slots_b.sum(dim=1))
            had_any_valid_b_steps.append(valid_slots_b.any(dim=1))

            inp_b = query_vec.unsqueeze(2)
            ctx_b = self.attention_linear_encoder(encoder_outputs_b.permute(0, 2, 1))
            attn_scores_b = torch.bmm(V_exp, torch.tanh(inp_b + ctx_b)).squeeze(1)
            attn_scores_b = attn_scores_b.masked_fill(mask_ptr_b, mask_fill_val)

            if debug:
                # Capture step-level debug info for the first N samples
                n_dbg = min(debug_n_samples, batch_size)
                for bi in range(n_dbg):
                    la_eff = int(eff_len_a[bi].item())
                    lb_eff = int(eff_len_b[bi].item())
                    ca = int(current_bA[bi].item())
                    cb = int(current_bB[bi].item())

                    # Candidate indices for this step (positions, not token ids)
                    cand_a = torch.where(valid_slots_a[bi])[0]
                    cand_b = torch.where(valid_slots_b[bi])[0]

                    # Top-k among *masked* logits (may be -inf if no valid)
                    def _topk_positions(scores_row, k):
                        if scores_row.numel() == 0:
                            return []
                        kk = min(k, scores_row.numel())
                        vals, idxs = torch.topk(scores_row, kk)
                        return [(int(idxs[j].item()), float(vals[j].item())) for j in range(kk)]

                    topk_a = _topk_positions(attn_scores_a[bi], debug_topk)
                    topk_b = _topk_positions(attn_scores_b[bi], debug_topk)

                    # Convert positions -> token strings for readability
                    def _pos_to_tok(seq_ids, pos):
                        try:
                            tok_id = int(seq_ids[pos].item())
                        except Exception:
                            tok_id = int(seq_ids[pos])
                        try:
                            return self.tokenizer.convert_ids_to_tokens(tok_id)
                        except Exception:
                            return str(tok_id)

                    ids_a_row = td["input_ids_a"][bi]
                    ids_b_row = td["input_ids_b"][bi]
                    topk_a_tok = [(p, _pos_to_tok(ids_a_row, p), s) for (p, s) in topk_a]
                    topk_b_tok = [(p, _pos_to_tok(ids_b_row, p), s) for (p, s) in topk_b]

                    rec = {
                        "step": int(step),
                        "sample": int(bi),
                        "current_boundary_a": ca,
                        "current_boundary_b": cb,
                        "eff_len_a": la_eff,
                        "eff_len_b": lb_eff,
                        "n_valid_a": int(cand_a.numel()),
                        "n_valid_b": int(cand_b.numel()),
                        "topk_a": topk_a_tok,
                        "topk_b": topk_b_tok,
                    }
                    debug_records.append(rec)

            # === 采样/Argmax ===
            pointer_a = torch.zeros(batch_size, dtype=torch.long, device=dev)
            pointer_b = torch.zeros(batch_size, dtype=torch.long, device=dev)
            logp_a = torch.zeros(batch_size, device=dev)
            logp_b = torch.zeros(batch_size, device=dev)
            
            has_valid_a = valid_slots_a.any(dim=1)
            has_valid_b = valid_slots_b.any(dim=1)

            # 处理 A
            if has_valid_a.any():
                rows = has_valid_a
                if decode_type == "sampling":
                    dist = Categorical(logits=attn_scores_a[rows])
                    act = dist.sample()
                    pointer_a[rows] = act
                    logp_a[rows] = dist.log_prob(act)
                else:
                    pointer_a[rows] = torch.argmax(attn_scores_a[rows], dim=1)

            if (~has_valid_a).any():
                rows = ~has_valid_a
                # Fall back to the last *non-special* token (avoid trailing [SEP]/[EOS])
                pointer_a[rows] = (eff_len_a[rows] - 1).clamp(min=1)
             

            # 处理 B
            if has_valid_b.any():
                rows = has_valid_b
                if decode_type == "sampling":
                    dist = Categorical(logits=attn_scores_b[rows])
                    act = dist.sample()
                    pointer_b[rows] = act
                    logp_b[rows] = dist.log_prob(act)
                else:
                    pointer_b[rows] = torch.argmax(attn_scores_b[rows], dim=1)
            
            if (~has_valid_b).any():
                rows = ~has_valid_b
                pointer_b[rows] = (eff_len_b[rows] - 1).clamp(min=1)

            if debug:
                n_dbg = min(debug_n_samples, batch_size)
                for bi in range(n_dbg):
                    pa = int(pointer_a[bi].item())
                    pb = int(pointer_b[bi].item())
                    ids_a_row = td["input_ids_a"][bi]
                    ids_b_row = td["input_ids_b"][bi]

                    def _pos_to_tok(seq_ids, pos):
                        try:
                            tok_id = int(seq_ids[pos].item())
                        except Exception:
                            tok_id = int(seq_ids[pos])
                        try:
                            return self.tokenizer.convert_ids_to_tokens(tok_id)
                        except Exception:
                            return str(tok_id)

                    # This matches our semantics: fallback when no valid slots
                    fallback_a = bool((~has_valid_a)[bi].item())
                    fallback_b = bool((~has_valid_b)[bi].item())
                    rec = {
                        "step": int(step),
                        "sample": int(bi),
                        "chosen_a_pos": pa,
                        "chosen_b_pos": pb,
                        "chosen_a_tok": _pos_to_tok(ids_a_row, pa),
                        "chosen_b_tok": _pos_to_tok(ids_b_row, pb),
                        "fallback_a": fallback_a,
                        "fallback_b": fallback_b,
                        "chosen_is_valid_a": bool(valid_slots_a[bi, pa].item()) if pa < valid_slots_a.size(1) else False,
                        "chosen_is_valid_b": bool(valid_slots_b[bi, pb].item()) if pb < valid_slots_b.size(1) else False,
                    }
                    debug_records.append(rec)

            pointers.append(torch.stack([pointer_a, pointer_b], dim=1))
            log_probs.append(logp_a + logp_b)

        
            feedback_emb_a_list = []
            
            input_ids_a = td['input_ids_a']
            lm_device = next(self.lm.parameters()).device

            for b in range(batch_size):
               
                s_a = current_bA[b].item() # 上一个边界 (包含在上一个片段中)
                e_a = pointer_a[b].item()  # 当前边界 (包含在当前片段中)
                
           
                real_start_a = s_a + 1 if s_a > 0 else 1  
                
              
                real_end_a = e_a + 1
           
                if real_end_a <= real_start_a:
                
                    seg_ids_a = torch.tensor([[self.tokenizer.pad_token_id]], device=lm_device)
                else:
                  
                    seg_ids_a = input_ids_a[b, real_start_a : real_end_a].unsqueeze(0).to(lm_device)

                # 计算 Mask (非 Pad 为 1)
                attn_mask_a_seg = (seg_ids_a != self.tokenizer.pad_token_id).long()
                
                # 调用 LM 提取特征
                with torch.no_grad():
                 
                    if attn_mask_a_seg.sum() == 0:
                        emb_val = torch.zeros(1, self.embedding_model.model.config.hidden_size, device=lm_device)
                    else:
                        outputs = self.lm(seg_ids_a, attention_mask=attn_mask_a_seg)
                       
                        emb_val = outputs.last_hidden_state.mean(dim=1)
                
                feedback_emb_a_list.append(emb_val.squeeze(0))

            # Stack 并放回 Policy 的 device 进行下一步投影
            feedback_tensor_a = torch.stack(feedback_emb_a_list, dim=0).to(self.device)
            
            # 投影作为 LSTM 下一步输入
            decoder_input = self.input_proj(feedback_tensor_a)

            # 更新状态
            current_bA = pointer_a
            current_bB = pointer_b

     
        actions = torch.stack(pointers, 1).view(batch_size, -1)
        log_likelihood = torch.stack(log_probs, 1).sum(dim=1)

        info = {}
        try:
            # [batch, steps] -> scalar stats
            vca = torch.stack(valid_counts_a_steps, dim=1).float()
            vcb = torch.stack(valid_counts_b_steps, dim=1).float()
            hva = torch.stack(had_any_valid_a_steps, dim=1)
            hvb = torch.stack(had_any_valid_b_steps, dim=1)

            # These are high-signal indicators:
            # - If avg_valid_* ~ 1.0 and frac_any_valid_* ~ 1.0, then logp tends to 0 => loss ~ 0
            # - If frac_any_valid_* is low, you fall back to (length-1) often => logp still ~ 0
            info.update(
                {
                    "avg_valid_a": vca.mean().detach(),
                    "avg_valid_b": vcb.mean().detach(),
                    "min_valid_a": vca.min().detach(),
                    "min_valid_b": vcb.min().detach(),
                    "frac_any_valid_a": hva.float().mean().detach(),
                    "frac_any_valid_b": hvb.float().mean().detach(),
                    "mean_log_likelihood": log_likelihood.detach().mean(),
                    "frac_zero_log_likelihood": (log_likelihood.detach().abs() < 1e-8).float().mean(),
                }
            )
        except Exception:
            # Never break training due to debug stats
            pass

        if debug:
            info["debug_records"] = debug_records
            if debug_print:
                # Print a compact summary (keeps stdout readable)
                for r in debug_records:
                    if "chosen_a_pos" in r:
                        print(
                            f"[DBG] step={r['step']} sample={r['sample']} "
                            f"A: pos={r['chosen_a_pos']} tok={r['chosen_a_tok']} valid={r['chosen_is_valid_a']} fallback={r['fallback_a']} | "
                            f"B: pos={r['chosen_b_pos']} tok={r['chosen_b_tok']} valid={r['chosen_is_valid_b']} fallback={r['fallback_b']}"
                        )
            if debug_log_path:
                try:
                    import json
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        for r in debug_records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                except Exception as e:
                    # Never break training/inspection due to logging failure
                    info["debug_log_error"] = str(e)

        return actions, log_likelihood, info
