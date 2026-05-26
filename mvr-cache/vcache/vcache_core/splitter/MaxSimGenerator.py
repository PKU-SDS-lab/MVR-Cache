import json

import numpy as np
import torch
from pathlib import Path
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator

from .embedding_model import EmbeddingModel

class MaxSimGenerator(Generator):
    def __init__(
        self,
        prompts=None,
        *,
        pairs=None,
        pairs_path: str | None = None,
        max_len=128,
        embedding_model=None,
        lm_model_name_or_path=None,
        seed: int | None = None,
    ):
        super().__init__()
        self.seed = seed

        # Mode A: prompts (list[str])
        # Mode B: pairs (list[dict] with sentence_1/sentence_2/correct)
        if pairs_path is not None and pairs is None:
            p = Path(pairs_path)
            with p.open("r", encoding="utf-8") as f:
                pairs = json.load(f)

        self.pairs = pairs
        self.prompts = prompts

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

        self.max_len = max_len
        
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
          
            self.embedding_model = EmbeddingModel(model_name=lm_model_name_or_path)

        self.embedding_model.model.eval()
     
        self.tokenizer = self.embedding_model.tokenizer
        self.lm = self.embedding_model.model

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
        if self.pairs_mode:
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

            texts_a = [self.pairs[i]["sentence_1"] for i in idxs]
            texts_b = [self.pairs[i]["sentence_2"] for i in idxs]
            correct = torch.tensor([int(self.pairs[i]["correct"]) for i in idxs], dtype=torch.float32)
        else:
            # 随机采样文本对 (with replacement)
            indices_a = np.random.randint(0, self.num_prompts, size=bs)
            indices_b = np.random.randint(0, self.num_prompts, size=bs)
            texts_a = [self.prompts[i] for i in indices_a]
            texts_b = [self.prompts[i] for i in indices_b]

        # ==================================================================
        #查看原始文本
        # print(f"--- [DEBUG] Batch Size: {bs}, Device: {device} ---")
        # for i in range(bs):
        #     print(f"  Pair {i+1}: A='{texts_a[i]}', B='{texts_b[i]}'")
        # print("----------------------------------------------------")
        # ==================================================================

        # 使用 EmbeddingModel 获取token级别的嵌入
        # Compute on `device` (often GPU) but return embeddings on CPU to avoid OOM when RL4CO
        # pre-generates large datasets (thousands of samples).
        token_embeds_a = self.embedding_model.get_token_embeddings(
            texts_a, max_length=self.max_len, device=device, return_device=torch.device("cpu")
        )
        token_embeds_b = self.embedding_model.get_token_embeddings(
            texts_b, max_length=self.max_len, device=device, return_device=torch.device("cpu")
        )
        
        embeddings_a = token_embeds_a['last_hidden_state']
        mask_a = token_embeds_a['attention_mask'].to(torch.bool)
        embeddings_b = token_embeds_b['last_hidden_state']
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
            device=torch.device("cpu"),
        )
        if correct is not None:
            td["correct"] = correct
        return td