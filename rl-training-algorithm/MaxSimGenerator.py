import torch
import numpy as np
import sys
import os
import json
import re
from pathlib import Path
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_model import EmbeddingModel
from similarity_evaluator import (
    IdSetSimilarityEvaluator,
    StringComparisonSimilarityEvaluator,
    choose_id_set_column,
    has_usable_ids,
)

def _clean_prompt_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*>\s*", "", s)
    return s.strip()

class MaxSimGenerator(Generator):
    def __init__(
        self,
        prompts=None,
        *,
        pairs=None,
        pairs_path: str | None = None,
        parquet_path: str | None = None,
        parquet_text_column: str = "prompt",
        label_mode: str = "none",  # none|auto|id_set|string
        response_column: str | None = None,
        return_row_indices: bool = False,
        precompute_token_embeddings: bool = False,
        sampling_mode: str = "pairs",
        nn_warmup_epochs: int = 5,
        nn_candidate_topk: int = 50,
        nn_full_pairwise: bool = False,
        nn_similarity_mode: str = "blockwise",  # blockwise|full
        nn_similarity_dtype: str = "float16",   # float16|float32 (only for mode=full)
        nn_rebuild_every_n_epochs: int = 1,
        nn_label_strategy: str = "skip",  # skip|zero
        max_len=128,
        embedding_model=None,
        lm_model_name_or_path=None,
        seed: int | None = None,
    ):
        super().__init__()
        self.seed = seed
        # Persistent RNG for sampling so batches don't repeat deterministically.
        # (Do NOT re-create RNG inside _generate, otherwise every batch is identical.)
        self._sample_rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()
        self.sampling_mode = str(sampling_mode).lower().strip()
        self.parquet_path = parquet_path
        self.parquet_text_column = str(parquet_text_column)
        self.label_mode = str(label_mode).lower().strip()
        self.response_column = response_column
        self.return_row_indices = bool(return_row_indices)
        self.precompute_token_embeddings = bool(precompute_token_embeddings)
        self.nn_warmup_epochs = int(nn_warmup_epochs)
        self.nn_candidate_topk = int(nn_candidate_topk)
        self.nn_full_pairwise = bool(nn_full_pairwise)
        self.nn_similarity_mode = str(nn_similarity_mode).lower().strip()
        self.nn_similarity_dtype = str(nn_similarity_dtype).lower().strip()
        if self.nn_similarity_mode not in {"blockwise", "full"}:
            raise ValueError(
                f"Unsupported nn_similarity_mode={nn_similarity_mode!r}. Use 'blockwise' or 'full'."
            )
        if self.nn_similarity_dtype not in {"float16", "float32"}:
            raise ValueError(
                f"Unsupported nn_similarity_dtype={nn_similarity_dtype!r}. Use 'float16' or 'float32'."
            )
        self.nn_rebuild_every_n_epochs = max(1, int(nn_rebuild_every_n_epochs))
        self.nn_label_strategy = str(nn_label_strategy).lower().strip()
        if self.nn_label_strategy not in {"skip", "zero"}:
            raise ValueError(f"Unsupported nn_label_strategy={nn_label_strategy!r}. Use 'skip' or 'zero'.")

        # Mode A: prompts (list[str]) OR parquet (single-column text + other fields)
        # Mode B: pairs (list[dict] with sentence_1/sentence_2/correct)
        if pairs_path is not None and pairs is None:
            p = Path(pairs_path)
            with p.open("r", encoding="utf-8") as f:
                pairs = json.load(f)

        self.pairs = pairs
        self.prompts = prompts

        # Parquet prompts mode: load prompts from parquet and keep the table (so other fields are available by row index)
        self._parquet_table = None
        self._parquet_id_col = None
        self._parquet_id_vals = None  # list[int|None]
        self._parquet_resp_col = None
        self._parquet_resp_vals = None  # list[str|None]
        self._label_evaluator = None
        if self.parquet_path is not None and self.pairs is None and self.prompts is None:
            try:
                import pyarrow.parquet as pq
            except Exception as e:
                raise RuntimeError(
                    "parquet_path was provided but pyarrow is not available. "
                    "Run in an environment with pyarrow installed."
                ) from e
            pf = pq.ParquetFile(self.parquet_path)
            self._parquet_table = pq.read_table(self.parquet_path)
            if self.parquet_text_column not in self._parquet_table.column_names:
                raise ValueError(
                    f"Parquet missing text column {self.parquet_text_column!r}. "
                    f"Available columns: {self._parquet_table.column_names}"
                )
            col = self._parquet_table[self.parquet_text_column]
            # Convert to python list[str]
            self.prompts = [ _clean_prompt_text(x.as_py()) for x in col ]

            # Decide label mode / evaluator (optional)
            cols = list(self._parquet_table.column_names)
            mode = self.label_mode
            if mode not in {"none", "auto", "id_set", "string"}:
                raise ValueError(f"Unsupported label_mode={self.label_mode!r}. Use none|auto|id_set|string.")

            if mode == "auto":
                id_col = choose_id_set_column(cols)
                if id_col is not None:
                    id_vals = self._parquet_table[id_col].to_pylist()
                    if has_usable_ids(id_vals):
                        mode = "id_set"
                        self._parquet_id_col = id_col
                        self._parquet_id_vals = id_vals
                if mode == "auto":
                    # Fall back to string mode only if response_column provided
                    if self.response_column is not None:
                        mode = "string"
                    else:
                        mode = "none"

            if mode == "id_set":
                id_col = choose_id_set_column(cols)
                if id_col is None:
                    raise ValueError("label_mode='id_set' but no ID_Set/id_set column exists in parquet.")
                self._parquet_id_col = id_col
                self._parquet_id_vals = self._parquet_table[id_col].to_pylist()
                self._label_evaluator = IdSetSimilarityEvaluator()

            if mode == "string":
                if self.response_column is None:
                    raise ValueError("label_mode='string' requires --response_column (parquet column name).")
                if self.response_column not in cols:
                    raise ValueError(
                        f"response_column={self.response_column!r} not found in parquet. Available: {cols}"
                    )
                self._parquet_resp_col = self.response_column
                self._parquet_resp_vals = self._parquet_table[self.response_column].to_pylist()
                self._label_evaluator = StringComparisonSimilarityEvaluator()

            # Log chosen labeling mode
            try:
                chosen = mode
                print(
                    f"[GEN][parquet] loaded rows={pf.metadata.num_rows} text_col={self.parquet_text_column!r} "
                    f"label_mode={chosen} id_col={self._parquet_id_col!r} resp_col={self._parquet_resp_col!r}"
                )
            except Exception:
                pass

        if self.pairs is not None:
            if not isinstance(self.pairs, list) or len(self.pairs) == 0:
                raise ValueError("pairs must be a non-empty list")
            self.pairs_mode = True
            self.num_pairs = len(self.pairs)
            self._pair_cursor = 0
            self._pair_order = np.arange(self.num_pairs)
            if self.seed is not None:
                self._pair_rng = np.random.default_rng(self.seed)
                self._pair_rng.shuffle(self._pair_order)
            else:
                np.random.shuffle(self._pair_order)
        else:
            if self.prompts is None:
                raise ValueError("Either prompts or pairs/pairs_path must be provided")
            if not isinstance(self.prompts, list) or len(self.prompts) == 0:
                raise ValueError("prompts must be a non-empty list")
            self.pairs_mode = False
            self.num_prompts = len(self.prompts)

        # Robustness: if we don't have `pairs`, sampling_mode="pairs" cannot work.
        # This happens easily for val/test generators where caller doesn't pass sampling_mode.
        if not self.pairs_mode and self.sampling_mode == "pairs":
            self.sampling_mode = "random"

        self.max_len = max_len
        
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
          
            self.embedding_model = EmbeddingModel(model_name=lm_model_name_or_path)

        self.embedding_model.model.eval()
     
        self.tokenizer = self.embedding_model.tokenizer
        self.lm = self.embedding_model.model

        # ------------------------------------------------------------
        # Optional in-RAM cache for token-level embeddings
        # ------------------------------------------------------------
        self._tok_cache_ready = False
        self._tok_cache_input_ids = None          # torch.LongTensor [N, L]
        self._tok_cache_attention_mask = None     # torch.BoolTensor [N, L]
        self._tok_cache_last_hidden_state = None  # torch.(Float/Half)Tensor [N, L, H] on CPU
        self._tok_cache_universe = None           # list[str] used for cache (alignment)
        self._tok_cache_prompt_to_idx = None      # dict[str,int] for mapping (pairs mode)

        # ----------------------------
        # Anchor / reverse-NN sampling state (optional)
        # ----------------------------
        self._epoch = 0
        self._nn_ready = False
        self._nn_last_built_epoch = None
        self._all_prompts = None  # list[str]
        self._prompt_to_idx = None  # dict[str,int]
        self._pair_label = None  # dict[(str,str), int]
        self._emb_all = None  # torch.Tensor [N,D] normalized on CPU
        self._nn_forward = None  # torch.LongTensor [N] : nn_idx[y] = x
        self._reverse_lists = None  # list[list[int]]: reverse_lists[x] = [y...]

        if self.sampling_mode == "anchor_nn":
            # Build prompt universe from either (a) pairs JSON or (b) parquet prompt column.
            # Labels can come from (a) pairs JSON lookup OR (b) parquet evaluator (id_set/string).
            if self.pairs is None and self._parquet_table is None:
                raise ValueError(
                    "sampling_mode='anchor_nn' requires either `pairs` (JSON pairs with labels) or `parquet_path`."
                )
            if self.pairs is not None:
                uniq = []
                seen = set()
                for ex in self.pairs:
                    a = _clean_prompt_text(ex.get("sentence_1", ""))
                    b = _clean_prompt_text(ex.get("sentence_2", ""))
                    if a and a not in seen:
                        seen.add(a); uniq.append(a)
                    if b and b not in seen:
                        seen.add(b); uniq.append(b)
                self._all_prompts = uniq
            else:
                # Parquet prompt universe
                self._all_prompts = list(self.prompts)
            self._prompt_to_idx = {p: i for i, p in enumerate(self._all_prompts)}
            # Label lookup for BCE (store both directions) if pairs provided
            self._pair_label = {}
            if self.pairs is not None:
                for ex in self.pairs:
                    a = _clean_prompt_text(ex["sentence_1"])
                    b = _clean_prompt_text(ex["sentence_2"])
                    try:
                        c = int(ex.get("correct", 0))
                    except Exception:
                        c = 0
                    self._pair_label[(a, b)] = c
                    self._pair_label[(b, a)] = c
            # Precompute whole-sentence embeddings for warmup NN + candidate retrieval
            try:
                # Keep embeddings on CPU to avoid holding NxD on GPU.
                emb = self.embedding_model.encode(self._all_prompts, convert_to_tensor=True, device=None)
                if not torch.is_tensor(emb):
                    emb = torch.tensor(emb)
                emb = emb.float().cpu()
                self._emb_all = torch.nn.functional.normalize(emb, p=2, dim=-1)
            except Exception as e:
                raise RuntimeError(f"Failed to precompute embeddings for anchor_nn sampling: {e}")

            # Build initial NN index (warmup mode uses embeddings only)
            self.rebuild_nn_index(epoch=0, policy=None, env=None)
        elif self.sampling_mode not in {"pairs", "random"}:
            raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}. Use 'pairs', 'random', or 'anchor_nn'.")

        # If requested, precompute token embeddings for the full prompt universe (CPU RAM).
        # This speeds up sampling by avoiding LM forward passes per batch.
        if self.precompute_token_embeddings:
            try:
                self._build_token_cache()
            except Exception as e:
                raise RuntimeError(f"Failed to precompute token embeddings cache: {e}")

    def _estimate_cache_bytes(self, *, n: int, l: int, h: int, dtype: torch.dtype) -> int:
        bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        # last_hidden_state dominates
        return int(n * l * h * bytes_per)

    def _build_token_cache(self):
        # Decide which prompt universe to cache:
        # - anchor_nn: cache the universe used for NN sampling (self._all_prompts)
        # - random/parquet prompts: cache self.prompts
        universe = None
        if getattr(self, "_all_prompts", None) is not None:
            universe = list(self._all_prompts)
        elif self.prompts is not None:
            universe = list(self.prompts)
        else:
            # pairs-only mode without an explicit universe; nothing sensible to cache
            print("[GEN][tok_cache] precompute requested but no prompt universe available; skipping.")
            return

        # Heuristic: store hidden states as float16 to reduce memory.
        # IMPORTANT: keep the cache on the LM device (usually GPU) so we avoid CPU->GPU copies per step.
        cache_dtype = torch.float16
        try:
            # Try to infer hidden size from model config
            h = int(getattr(getattr(self.lm, "config", None), "hidden_size", 768))
        except Exception:
            h = 768
        n = len(universe)
        l = int(self.max_len)
        est = self._estimate_cache_bytes(n=n, l=l, h=h, dtype=cache_dtype)
        try:
            est_gb = est / (1024**3)
            print(f"[GEN][tok_cache] building token cache: N={n} L={l} H~={h} dtype={str(cache_dtype).replace('torch.', '')} est_hidden_mem≈{est_gb:.2f}GB")
        except Exception:
            pass

        # Compute in batches on LM device and keep the cache on that device.
        # NOTE: This is intentionally simple; no user-exposed batch size per request.
        bs = 64
        device = next(self.lm.parameters()).device
        all_input_ids = []
        all_attn = []
        all_hid = []
        for start in range(0, n, bs):
            chunk = universe[start:start + bs]
            out = self.embedding_model.get_token_embeddings(
                chunk, max_length=self.max_len, device=device, return_device=device
            )
            # Standard keys from EmbeddingModel.get_token_embeddings
            hid = out["last_hidden_state"].to(dtype=cache_dtype, copy=False)  # [b,L,H] on device
            attn = out["attention_mask"].to(torch.bool, copy=False)          # [b,L] on device
            ids = out["input_ids"].to(torch.long, copy=False)                # [b,L] on device
            all_hid.append(hid)
            all_attn.append(attn)
            all_input_ids.append(ids)

        self._tok_cache_last_hidden_state = torch.cat(all_hid, dim=0).contiguous()
        self._tok_cache_attention_mask = torch.cat(all_attn, dim=0).contiguous()
        self._tok_cache_input_ids = torch.cat(all_input_ids, dim=0).contiguous()
        self._tok_cache_universe = universe
        # Mapping helps when pairs-mode provides texts; we can map back to cached indices when possible.
        self._tok_cache_prompt_to_idx = {p: i for i, p in enumerate(universe)}
        self._tok_cache_ready = True
        print(f"[GEN][tok_cache] ready: input_ids={tuple(self._tok_cache_input_ids.shape)} last_hidden_state={tuple(self._tok_cache_last_hidden_state.shape)}")

    def _token_batch_from_cache(self, indices: np.ndarray | list[int] | torch.Tensor) -> dict:
        if not self._tok_cache_ready:
            raise RuntimeError("token cache not ready")
        if isinstance(indices, torch.Tensor):
            idx = indices.detach().long()
        else:
            idx = torch.tensor(list(indices), dtype=torch.long, device=self._tok_cache_input_ids.device)
        ids = self._tok_cache_input_ids.index_select(0, idx)
        attn = self._tok_cache_attention_mask.index_select(0, idx)
        hid = self._tok_cache_last_hidden_state.index_select(0, idx)
        return {"input_ids": ids, "attention_mask": attn, "last_hidden_state": hid}

    def _pair_correct_by_indices(self, ia: int, ib: int) -> int | None:
        # 1) pairs JSON lookup by string (if available)
        if self._pair_label is not None and self._tok_cache_universe is not None:
            try:
                a = self._tok_cache_universe[int(ia)]
                b = self._tok_cache_universe[int(ib)]
                return self._pair_label.get((a, b), None)
            except Exception:
                return None
        # 2) parquet label evaluator by row fields (if available)
        if self._parquet_table is not None and self._label_evaluator is not None:
            if isinstance(self._label_evaluator, IdSetSimilarityEvaluator):
                id_a = self._parquet_id_vals[int(ia)] if self._parquet_id_vals is not None else None
                id_b = self._parquet_id_vals[int(ib)] if self._parquet_id_vals is not None else None
                return 1 if self._label_evaluator.answers_similar("", "", id_set_a=id_a, id_set_b=id_b) else 0
            if isinstance(self._label_evaluator, StringComparisonSimilarityEvaluator):
                ra = self._parquet_resp_vals[int(ia)] if self._parquet_resp_vals is not None else None
                rb = self._parquet_resp_vals[int(ib)] if self._parquet_resp_vals is not None else None
                return 1 if self._label_evaluator.answers_similar(ra, rb) else 0
        return None

    def set_epoch(self, epoch: int, *, policy=None, env=None):
        """Called by a Lightning callback to let the generator refresh epoch-dependent sampling."""
        self._epoch = int(epoch)
        # Rebuild on schedule (and only if in anchor_nn mode)
        if self.sampling_mode != "anchor_nn":
            return
        if self._nn_last_built_epoch is None:
            self.rebuild_nn_index(epoch=self._epoch, policy=policy, env=env)
            return
        if self._epoch == self._nn_last_built_epoch:
            return
        if self._epoch % self.nn_rebuild_every_n_epochs != 0:
            return
        self.rebuild_nn_index(epoch=self._epoch, policy=policy, env=env)

    def _pair_correct(self, a: str, b: str):
        # 1) If we have explicit labels (pairs JSON), use them
        if self._pair_label is not None and len(self._pair_label) > 0:
            v = self._pair_label.get((a, b), None)
            if v is not None:
                return v
        # 2) If we have parquet evaluator, compute label from row fields
        if self._label_evaluator is None or self._prompt_to_idx is None:
            return None
        ia = self._prompt_to_idx.get(a, None)
        ib = self._prompt_to_idx.get(b, None)
        if ia is None or ib is None:
            return None
        if isinstance(self._label_evaluator, IdSetSimilarityEvaluator):
            id_a = self._parquet_id_vals[ia] if self._parquet_id_vals is not None else None
            id_b = self._parquet_id_vals[ib] if self._parquet_id_vals is not None else None
            return 1 if self._label_evaluator.answers_similar("", "", id_set_a=id_a, id_set_b=id_b) else 0
        if isinstance(self._label_evaluator, StringComparisonSimilarityEvaluator):
            ra = self._parquet_resp_vals[ia] if self._parquet_resp_vals is not None else None
            rb = self._parquet_resp_vals[ib] if self._parquet_resp_vals is not None else None
            return 1 if self._label_evaluator.answers_similar(ra, rb) else 0
        return None

    def rebuild_nn_index(self, *, epoch: int, policy=None, env=None):
        """Build forward NN mapping y->x, then reverse lists for anchor sampling.

        Warmup (<nn_warmup_epochs): use whole-sentence embedding cosine NN.
        After warmup: use MaxSim NN (policy segmentation + env reward) over candidates.
        """
        if self.sampling_mode != "anchor_nn":
            return
        epoch = int(epoch)
        if self._all_prompts is None or self._emb_all is None:
            raise RuntimeError("anchor_nn sampling requested but prompt embeddings are not initialized.")
        N = len(self._all_prompts)
        if N < 2:
            raise RuntimeError("Need at least 2 prompts for anchor_nn sampling.")

        def _full_similarity_matrix_topk(
            emb: torch.Tensor,
            *,
            k: int,
            dtype: str,
        ) -> torch.LongTensor:
            """
            Build the full NxN similarity matrix in RAM (cosine via dot on normalized vectors),
            exclude diagonal, and return top-k indices [N,k].

            WARNING: O(N^2) memory. For N=30k:
              - float32: ~3.6GB
              - float16: ~1.8GB
            """
            n = int(emb.shape[0])
            kk = int(max(1, min(k, n - 1)))
            # Prefer GPU for the NxN matmul when available (much faster for N~3k),
            # but keep the result indices on CPU.
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            emb2 = emb.to(dev, non_blocking=True)
            if dtype == "float16":
                emb2 = emb2.to(dtype=torch.float16)
            else:
                emb2 = emb2.to(dtype=torch.float32)
            sim = emb2 @ emb2.T  # [N,N]
            # Exclude self
            sim.fill_diagonal_(-1e9)
            return torch.topk(sim, k=kk, dim=1).indices.to("cpu")

        def _topk_blockwise(
            emb: torch.Tensor,
            *,
            k: int,
            block_rows: int = 256,
            block_cols: int = 4096,
        ) -> torch.LongTensor:
            """
            Compute top-k cosine neighbors for each row of `emb` WITHOUT building the full NxN matrix.

            Assumes `emb` is L2-normalized on CPU (so dot == cosine similarity).
            Returns indices [N, k] on CPU, excluding self.
            """
            if emb.dim() != 2:
                raise ValueError(f"Expected emb [N,D], got shape={tuple(emb.shape)}")
            n = int(emb.shape[0])
            kk = int(max(1, min(k, n - 1)))
            out = torch.empty((n, kk), dtype=torch.long, device="cpu")

            emb = emb.contiguous()

            for i0 in range(0, n, int(block_rows)):
                i1 = min(n, i0 + int(block_rows))
                q = emb[i0:i1]  # [Br,D]
                br = int(q.shape[0])

                best_vals = torch.full((br, kk), -1e9, dtype=torch.float32)
                best_idx = torch.full((br, kk), -1, dtype=torch.long)

                for j0 in range(0, n, int(block_cols)):
                    j1 = min(n, j0 + int(block_cols))
                    c = emb[j0:j1]  # [Bc,D]
                    scores = q @ c.T  # [Br,Bc]

                    # Exclude self if it falls inside this column block.
                    if not (j1 <= i0 or j0 >= i1):
                        for r in range(br):
                            gi = i0 + r
                            if j0 <= gi < j1:
                                scores[r, gi - j0] = -1e9

                    local_k = min(kk, int(scores.shape[1]))
                    local_vals, local_pos = torch.topk(scores, k=local_k, dim=1)
                    local_idx = local_pos.to(torch.long) + int(j0)

                    merged_vals = torch.cat([best_vals, local_vals], dim=1)
                    merged_idx = torch.cat([best_idx, local_idx], dim=1)
                    new_vals, new_pos = torch.topk(merged_vals, k=kk, dim=1)
                    new_idx = merged_idx.gather(1, new_pos)
                    best_vals, best_idx = new_vals, new_idx

                out[i0:i1] = best_idx

            return out

        # -------- Warmup: embedding cosine NN --------
        if epoch < self.nn_warmup_epochs or policy is None or env is None:
            if self.nn_similarity_mode == "full":
                top1 = _full_similarity_matrix_topk(
                    self._emb_all, k=1, dtype=self.nn_similarity_dtype
                )
            else:
                top1 = _topk_blockwise(self._emb_all, k=1)  # [N,1]
            self._nn_forward = top1[:, 0].long().cpu()
        else:
            # -------- After warmup: MaxSim NN over candidates --------
            # Candidate selection: either full pairwise (expensive) or top-k by embedding sim.
            # We score candidates with the current policy+env via env._get_reward(policy(td).actions).
            device = next(policy.parameters()).device
            max_segments = getattr(env, "max_segments", None)
            if max_segments is None:
                raise RuntimeError("env.max_segments not found; cannot compute MaxSim NN.")

            # Precompute top-k candidates by embedding sim (on CPU)
            emb = self._emb_all

            if self.nn_full_pairwise:
                cand_lists = [torch.tensor([j for j in range(N) if j != i], dtype=torch.long) for i in range(N)]
            else:
                k = max(1, min(self.nn_candidate_topk, N - 1))
                if self.nn_similarity_mode == "full":
                    topk = _full_similarity_matrix_topk(emb, k=k, dtype=self.nn_similarity_dtype)
                else:
                    topk = _topk_blockwise(emb, k=k)  # [N,k]
                cand_lists = [topk[i] for i in range(N)]

            nn_out = torch.empty(N, dtype=torch.long)
            policy.eval()
            # We batch candidate scoring per anchor to keep memory bounded.
            # This is still heavy; for N=2500 and k=50 it's 125k pair scores per rebuild.
            with torch.no_grad():
                for i in range(N):
                    cand = cand_lists[i]
                    if cand.numel() == 0:
                        nn_out[i] = (i + 1) % N
                        continue
                    # Build batch texts
                    a_txt = self._all_prompts[i]
                    b_txts = [self._all_prompts[int(j)] for j in cand.tolist()]
                    a_txts = [a_txt] * len(b_txts)

                    # Token embeddings on the policy device to avoid repeated CPU->GPU copies
                    token_embeds_a = self.embedding_model.get_token_embeddings(
                        a_txts, max_length=self.max_len, device=device, return_device=device
                    )
                    token_embeds_b = self.embedding_model.get_token_embeddings(
                        b_txts, max_length=self.max_len, device=device, return_device=device
                    )

                    td = TensorDict(
                        {
                            "token_embeddings_a": token_embeds_a["last_hidden_state"],
                            "attention_mask_a": token_embeds_a["attention_mask"].to(torch.bool),
                            "token_embeddings_b": token_embeds_b["last_hidden_state"],
                            "attention_mask_b": token_embeds_b["attention_mask"].to(torch.bool),
                            "input_ids_a": token_embeds_a["input_ids"],
                            "input_ids_b": token_embeds_b["input_ids"],
                            "length_a": token_embeds_a["attention_mask"].sum(dim=1),
                            "length_b": token_embeds_b["attention_mask"].sum(dim=1),
                        },
                        batch_size=[len(b_txts)],
                        device=device,
                    )

                    # Policy returns a full plan actions [B, 2*max_segments]
                    out = policy(td, env, phase="val", decode_type="greedy")
                    actions = out.get("actions", None)
                    if actions is None:
                        # RL4CO policies sometimes return action under "action"
                        actions = out.get("action", None)
                    if actions is None:
                        raise RuntimeError("Policy output missing 'actions'/'action'; cannot score MaxSim NN.")
                    # Score via env reward
                    rewards = env._get_reward(td, actions).detach().float().to("cpu")  # [B]
                    best_pos = int(torch.argmax(rewards).item())
                    nn_out[i] = int(cand[best_pos].item())

            self._nn_forward = nn_out.cpu()

        # Build reverse lists: for each x, collect y such that nn[y]==x
        reverse = [[] for _ in range(N)]
        for y in range(N):
            x = int(self._nn_forward[y].item())
            if 0 <= x < N and x != y:
                reverse[x].append(y)
        self._reverse_lists = reverse
        self._nn_ready = True
        self._nn_last_built_epoch = epoch
        try:
            mode = "embed" if epoch < self.nn_warmup_epochs else "maxsim"
            print(f"[GEN][anchor_nn] rebuilt NN index at epoch={epoch} mode={mode} N={N}")
        except Exception:
            pass

    def _generate(self, batch_size, **kwargs):
        
        if isinstance(batch_size, (list, tuple)):
            bs = int(batch_size[0])
        elif isinstance(batch_size, torch.Size):
            bs = int(batch_size[0])
        else:
            bs = int(batch_size)
            
      
        device = next(self.lm.parameters()).device
        if not hasattr(self, "_dbg_printed"):
            print(f"[GEN] bs={bs} device={device} lm_device={device} max_len={self.max_len}")
            self._dbg_printed = True

        # Sample text pairs
        correct = None
        if self.sampling_mode == "anchor_nn":
            if not self._nn_ready or self._reverse_lists is None:
                # Safety fallback: build embedding NN
                self.rebuild_nn_index(epoch=int(getattr(self, "_epoch", 0)), policy=None, env=None)
            N = len(self._all_prompts)
            texts_a = []
            texts_b = []
            idx_a_list = []
            idx_b_list = []
            correct_list = []

            # Fill bs pairs by repeatedly sampling anchors and their reverse-NN sets
            max_tries = 10_000
            tries = 0
            rng = getattr(self, "_sample_rng", None) or np.random.default_rng()
            while len(texts_a) < bs and tries < max_tries:
                tries += 1
                x = int(rng.integers(0, N))
                ys = self._reverse_lists[x] if self._reverse_lists is not None else []
                if not ys:
                    continue
                # Shuffle ys to avoid always taking the same ones
                ys_local = ys.copy()
                rng.shuffle(ys_local)
                for y in ys_local:
                    if len(texts_a) >= bs:
                        break
                    a_txt = self._all_prompts[x]
                    b_txt = self._all_prompts[int(y)]
                    lab = None
                    # If we can label by indices (parquet evaluator or pairs map), do it without string ops
                    try:
                        lab = self._pair_correct_by_indices(int(x), int(y))
                    except Exception:
                        lab = None
                    if lab is None:
                        lab = self._pair_correct(a_txt, b_txt)
                    if lab is None:
                        if self.nn_label_strategy == "skip":
                            continue
                        lab = 0
                    texts_a.append(a_txt)
                    texts_b.append(b_txt)
                    idx_a_list.append(int(x))
                    idx_b_list.append(int(y))
                    correct_list.append(int(lab))

            # Hard fallback: if we couldn't fill, fall back to existing pairs sampling
            if len(texts_a) < bs:
                remaining = bs - len(texts_a)
                # If we have explicit labeled pairs, cycle them without replacement
                if self.pairs is not None and getattr(self, "num_pairs", 0) > 0 and hasattr(self, "_pair_order"):
                    idxs = []
                    for _ in range(remaining):
                        if self._pair_cursor >= self.num_pairs:
                            self._pair_cursor = 0
                            if self.seed is not None:
                                self._pair_rng.shuffle(self._pair_order)
                            else:
                                np.random.shuffle(self._pair_order)
                        idxs.append(int(self._pair_order[self._pair_cursor]))
                        self._pair_cursor += 1
                    texts_a.extend([_clean_prompt_text(self.pairs[i]["sentence_1"]) for i in idxs])
                    texts_b.extend([_clean_prompt_text(self.pairs[i]["sentence_2"]) for i in idxs])
                    correct_list.extend([int(self.pairs[i].get("correct", 0)) for i in idxs])
                else:
                    # Parquet-only anchor_nn can legitimately have no `pairs`.
                    # Fall back to random prompt pairs and derive labels if possible.
                    ia = rng.integers(0, N, size=remaining)
                    ib = rng.integers(0, N, size=remaining)
                    for a_i, b_i in zip(ia.tolist(), ib.tolist()):
                        a_txt = self._all_prompts[int(a_i)]
                        b_txt = self._all_prompts[int(b_i)]
                        lab = None
                        try:
                            lab = self._pair_correct_by_indices(int(a_i), int(b_i))
                        except Exception:
                            lab = None
                        if lab is None:
                            lab = self._pair_correct(a_txt, b_txt)
                        if lab is None:
                            lab = 0
                        texts_a.append(a_txt)
                        texts_b.append(b_txt)
                        idx_a_list.append(int(a_i))
                        idx_b_list.append(int(b_i))
                        correct_list.append(int(lab))

            correct = torch.tensor(correct_list[:bs], dtype=torch.float32)
        elif self.pairs_mode or self.sampling_mode == "pairs":
            # Prefer sampling *without replacement* by cycling a shuffled order
            idxs = []
            for _ in range(bs):
                if self._pair_cursor >= self.num_pairs:
                    self._pair_cursor = 0
                    if self.seed is not None:
                        self._pair_rng.shuffle(self._pair_order)
                    else:
                        np.random.shuffle(self._pair_order)
                idxs.append(int(self._pair_order[self._pair_cursor]))
                self._pair_cursor += 1

            texts_a = [_clean_prompt_text(self.pairs[i]["sentence_1"]) for i in idxs]
            texts_b = [_clean_prompt_text(self.pairs[i]["sentence_2"]) for i in idxs]
            correct = torch.tensor([int(self.pairs[i].get("correct", 0)) for i in idxs], dtype=torch.float32)
        else:
            # Random prompt pairs (with replacement)
            rng = getattr(self, "_sample_rng", None) or np.random.default_rng()
            indices_a = rng.integers(0, self.num_prompts, size=bs)
            indices_b = rng.integers(0, self.num_prompts, size=bs)
            texts_a = [_clean_prompt_text(self.prompts[i]) for i in indices_a]
            texts_b = [_clean_prompt_text(self.prompts[i]) for i in indices_b]

            # Attach labels from parquet if configured
            if self._parquet_table is not None and self._label_evaluator is not None:
                labs = []
                if isinstance(self._label_evaluator, IdSetSimilarityEvaluator):
                    for ia, ib in zip(indices_a.tolist(), indices_b.tolist()):
                        id_a = self._parquet_id_vals[int(ia)] if self._parquet_id_vals is not None else None
                        id_b = self._parquet_id_vals[int(ib)] if self._parquet_id_vals is not None else None
                        labs.append(1 if self._label_evaluator.answers_similar("", "", id_set_a=id_a, id_set_b=id_b) else 0)
                elif isinstance(self._label_evaluator, StringComparisonSimilarityEvaluator):
                    for ia, ib in zip(indices_a.tolist(), indices_b.tolist()):
                        ra = self._parquet_resp_vals[int(ia)] if self._parquet_resp_vals is not None else None
                        rb = self._parquet_resp_vals[int(ib)] if self._parquet_resp_vals is not None else None
                        labs.append(1 if self._label_evaluator.answers_similar(ra, rb) else 0)
                correct = torch.tensor(labs, dtype=torch.float32)

        # ==================================================================
        #查看原始文本
        # print(f"--- [DEBUG] Batch Size: {bs}, Device: {device} ---")
        # for i in range(bs):
        #     print(f"  Pair {i+1}: A='{texts_a[i]}', B='{texts_b[i]}'")
        # print("----------------------------------------------------")
        # ==================================================================

        # Token-level embeddings:
        # - Default: compute on the fly with the LM (GPU) and return to CPU.
        # - If precompute_token_embeddings enabled: gather from in-RAM cache (CPU) by row indices.
        if self._tok_cache_ready:
            if self.sampling_mode == "anchor_nn":
                if len(idx_a_list) < bs or len(idx_b_list) < bs:
                    # Safety fallback
                    token_embeds_a = self.embedding_model.get_token_embeddings(
                        texts_a, max_length=self.max_len, device=device, return_device=torch.device("cpu")
                    )
                    token_embeds_b = self.embedding_model.get_token_embeddings(
                        texts_b, max_length=self.max_len, device=device, return_device=torch.device("cpu")
                    )
                else:
                    token_embeds_a = self._token_batch_from_cache(idx_a_list[:bs])
                    token_embeds_b = self._token_batch_from_cache(idx_b_list[:bs])
            elif (not self.pairs_mode) and self.sampling_mode != "pairs":
                token_embeds_a = self._token_batch_from_cache(indices_a.tolist())
                token_embeds_b = self._token_batch_from_cache(indices_b.tolist())
            else:
                # pairs-mode randomization uses free-form strings; fall back unless we can map them
                try:
                    if self._tok_cache_prompt_to_idx is not None:
                        ia = [self._tok_cache_prompt_to_idx.get(t, -1) for t in texts_a]
                        ib = [self._tok_cache_prompt_to_idx.get(t, -1) for t in texts_b]
                        if min(ia) >= 0 and min(ib) >= 0:
                            token_embeds_a = self._token_batch_from_cache(ia)
                            token_embeds_b = self._token_batch_from_cache(ib)
                        else:
                            raise RuntimeError("some pair texts not found in cache universe")
                    else:
                        raise RuntimeError("no cache mapping available")
                except Exception:
                    token_embeds_a = self.embedding_model.get_token_embeddings(
                        texts_a, max_length=self.max_len, device=device, return_device=torch.device("cpu")
                    )
                    token_embeds_b = self.embedding_model.get_token_embeddings(
                        texts_b, max_length=self.max_len, device=device, return_device=torch.device("cpu")
                    )
        else:
            # Compute on `device` (often GPU) but return embeddings on CPU to avoid OOM when RL4CO
            # pre-generates large datasets (thousands of samples).
            token_embeds_a = self.embedding_model.get_token_embeddings(
                texts_a, max_length=self.max_len, device=device, return_device=torch.device("cpu")
            )
            token_embeds_b = self.embedding_model.get_token_embeddings(
                texts_b, max_length=self.max_len, device=device, return_device=torch.device("cpu")
            )
        
        # Policy uses Conv1d/Linear layers whose params are float32 by default.
        # Ensure embeddings are float32 to avoid "Input type (Half) and bias type (float)" errors,
        # especially when using the in-RAM cache (which may store float16 to save RAM).
        embeddings_a = token_embeds_a['last_hidden_state'].to(dtype=torch.float32)
        mask_a = token_embeds_a['attention_mask'].to(torch.bool)
        embeddings_b = token_embeds_b['last_hidden_state'].to(dtype=torch.float32)
        mask_b = token_embeds_b['attention_mask'].to(torch.bool)
       
        inputs_a = {
            'input_ids': token_embeds_a['input_ids'],
            'attention_mask': token_embeds_a['attention_mask']
        }
        inputs_b = {
            'input_ids': token_embeds_b['input_ids'],
            'attention_mask': token_embeds_b['attention_mask']
        }
        
       
        lengths_a = inputs_a['attention_mask'].sum(dim=1)
        lengths_b = inputs_b['attention_mask'].sum(dim=1)

        td = TensorDict(
            {
               
               "token_embeddings_a": embeddings_a,
                "attention_mask_a": mask_a,  
                "token_embeddings_b": embeddings_b,
                "attention_mask_b": mask_b,

             
                "input_ids_a": inputs_a['input_ids'],
                "input_ids_b": inputs_b['input_ids'],
                "length_a": lengths_a,
                "length_b": lengths_b,
            },
            batch_size=[bs],
            # Keep TensorDict on the same device as embeddings to avoid CPU->GPU copies in env/policy/reward.
            device=embeddings_a.device,
        )
        # If requested, return row indices so the caller can look up other fields in the parquet table
        # (strings like responses are not stored in TensorDict; use these indices to index the table/df externally).
        if self.return_row_indices and not self.pairs_mode and self.sampling_mode != "anchor_nn":
            td["row_idx_a"] = torch.tensor(indices_a, dtype=torch.long)
            td["row_idx_b"] = torch.tensor(indices_b, dtype=torch.long)
        if correct is not None:
            td["correct"] = correct
        return td
