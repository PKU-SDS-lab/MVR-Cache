import os

# Allow users to opt into a mirror, but keep the release portable by default.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

from transformers import AutoModel, AutoTokenizer
import torch  

class EmbeddingModel:
    """ 计算文本的向量嵌入 """
    def __init__(self, model_name=None, device=None):
        if model_name is None:
            env_path = os.environ.get("BGE_MODEL_PATH")
            if env_path and os.path.isdir(env_path):
                model_name = os.path.normpath(env_path)
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                local_candidates = [
                    os.path.join(current_dir, "models", "BAAI__bge-base-en-v1.5"),
                    os.path.join(current_dir, "bge-base-en"),
                ]
                model_name = next(
                    (os.path.normpath(path) for path in local_candidates if os.path.isdir(path)),
                    "BAAI/bge-base-en-v1.5",
                )

       
        if os.path.isdir(model_name):
            tok_json = os.path.join(model_name, "tokenizer.json")
            if os.path.isfile(tok_json):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, tokenizer_file=tok_json, local_files_only=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        else:
            print(f"[INFO] Loading model {model_name} from {os.environ.get('HF_ENDPOINT')}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        # Move model to specified device if provided
        if device is not None:
            self.model.to(device)
            print(f"[DEVICE] EmbeddingModel moved to {device}")
        else:
            current_device = next(self.model.parameters()).device
            print(f"[DEVICE] EmbeddingModel loaded on {current_device}")

        self.model.eval()

    def get_embedding(self, text):
        """ 获取文本的向量嵌入 """
      
        device = next(self.model.parameters()).device
    
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hs = outputs.last_hidden_state  # [1, L, H]
        attn = inputs.get("attention_mask", None)
        if attn is None:
            pooled = hs.mean(dim=1)
        else:
            attn_f = attn.to(dtype=hs.dtype).unsqueeze(-1)
            pooled = (hs * attn_f).sum(dim=1) / attn_f.sum(dim=1).clamp_min(1.0)

        return pooled.squeeze().cpu().numpy()

    def get_embedding_tensor(self, text: str, device: torch.device | str | None = None) -> torch.Tensor:
        """
        Get a single embedding as a torch.Tensor on the requested device.

        Important: This preserves the *exact* pooling behavior of `get_embedding`:
        - tokenization uses padding=True, truncation=True
        - pooling is an unmasked mean over `last_hidden_state` (includes pad tokens if present)

        This method exists to avoid the GPU->CPU->GPU round-trip of `.cpu().numpy()`
        when the caller wants to keep downstream math on GPU (e.g., MaxSim).
        """
        if device is None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device(device)

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Ensure the model is on the same device as inputs (no-op if already there).
        # This mirrors `get_embedding` behavior where the model device is the source of truth.
        if next(self.model.parameters()).device != device:
            self.model.to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        hs = outputs.last_hidden_state  # [1, L, H]
        attn = inputs.get("attention_mask", None)
        if attn is None:
            pooled = hs.mean(dim=1)
        else:
            attn_f = attn.to(dtype=hs.dtype).unsqueeze(-1)
            pooled = (hs * attn_f).sum(dim=1) / attn_f.sum(dim=1).clamp_min(1.0)

        # Keep dtype/device for downstream similarity computation.
        return pooled.squeeze()

    def get_embeddings_tensor(
        self, texts: list[str], device: torch.device | str | None = None
    ) -> torch.Tensor:
        """
        Batched version of `get_embedding_tensor`.

        Semantics: matches what you'd get by calling `get_embedding_tensor` on each
        text independently (batch_size=1 => no padding) by using **masked mean pooling**
        over non-padding tokens when batching introduces padding.

        Returns:
            torch.Tensor of shape [len(texts), hidden_size] on `device`.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list[str]")

        if device is None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device(device)

        # Ensure model is on the target device.
        #
        # Important: `torch.device("cuda")` has `index=None`, while a model is typically
        # on `cuda:0`. Treat these as equivalent to avoid an expensive `.to(...)` call
        # on every invocation in hot paths.
        cur = next(self.model.parameters()).device
        same_device = (cur == device)
        if not same_device and cur.type == "cuda" and device.type == "cuda":
            # If caller asked for generic "cuda", accept the current cuda:<idx>.
            if device.index is None:
                same_device = True
            # If caller asked for a specific cuda:<idx>, require exact match.
        if not same_device:
            self.model.to(device)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hs = outputs.last_hidden_state  # [B, L, H]
        attn = inputs.get("attention_mask", None)
        if attn is None:
            # Fallback to unmasked mean (shouldn't happen for HF tokenizers)
            return hs.mean(dim=1)

        # Masked mean pooling: sum over real tokens / count(real tokens)
        # This reproduces the batch_size=1 behavior (no pad tokens).
        attn_f = attn.to(dtype=hs.dtype).unsqueeze(-1)  # [B, L, 1]
        summed = (hs * attn_f).sum(dim=1)  # [B, H]
        counts = attn_f.sum(dim=1).clamp_min(1.0)  # [B, 1]
        return summed / counts
    def get_token_embeddings(self, texts, max_length=None, device=None, return_device=None):
        """
        获取token级别的嵌入（用于指针网络输入）
        
        Args:
            texts: 字符串列表
            max_length: 最大长度，None则使用模型默认
            device: 指定设备
        
        Returns:
            dict: 包含 'last_hidden_state', 'input_ids', 'attention_mask'
        """
      
        batch_size_total = len(texts)
        if batch_size_total == 0:
            raise ValueError("texts 不能为空")

        chunk_size = min(8, max(1, batch_size_total // 2)) 

        hidden_states_list = []
        input_ids_list = []
        attention_mask_list = []

        # NOTE: When RL4CO builds datasets, it may call the generator with very large batch_size
        # (e.g., thousands). Keeping all token embeddings on GPU will OOM. To avoid this, we
        # optionally stream chunk outputs to `return_device` (often CPU) before concatenation.
        with torch.no_grad():
            for start_idx in range(0, batch_size_total, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size_total)
                texts_chunk = texts[start_idx:end_idx]

                if max_length is not None:
                    inputs = self.tokenizer(
                        texts_chunk, return_tensors="pt", padding='max_length',
                        truncation=True, max_length=max_length
                    )
                else:
                    inputs = self.tokenizer(texts_chunk, return_tensors="pt", padding=True, truncation=True)

                if device is not None:
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                if device is not None and torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                hs = outputs.last_hidden_state
                ids = inputs["input_ids"]
                attn = inputs.get("attention_mask", None)

                if return_device is not None:
                    hs = hs.to(return_device)
                    ids = ids.to(return_device)
                    if attn is not None:
                        attn = attn.to(return_device)

                hidden_states_list.append(hs)
                input_ids_list.append(ids)
                attention_mask_list.append(attn)

                
                del outputs, hs
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        last_hidden_state = torch.cat(hidden_states_list, dim=0)
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = None
        if all(m is not None for m in attention_mask_list):
            attention_mask = torch.cat(attention_mask_list, dim=0)

        return {
            'last_hidden_state': last_hidden_state,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def encode(self, texts, convert_to_tensor=False, device=None):
        """
        兼容SentenceTransformer的encode方法
        支持单个字符串或字符串列表
        
        Args:
            texts: 单个字符串或字符串列表
            convert_to_tensor: 是否返回tensor，默认False返回numpy数组
            device: 如果convert_to_tensor=True，指定tensor的设备
        
        Returns:
            如果convert_to_tensor=False: numpy数组 shape=(n, embedding_dim) 或 (embedding_dim,)
            如果convert_to_tensor=True: torch.Tensor shape=(n, embedding_dim) 或 (embedding_dim,)
        """
        # 处理单个字符串输入
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # 分块编码
        batch_size_total = len(texts)
        chunk_size = min(16, max(1, batch_size_total // 2))
        emb_list = []

        if device is not None:
            self.model.to(device)

        with torch.no_grad():
            for start_idx in range(0, batch_size_total, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size_total)
                texts_chunk = texts[start_idx:end_idx]
                inputs = self.tokenizer(texts_chunk, return_tensors="pt", padding=True, truncation=True)
                if device is not None:
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                if device is not None and torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                emb_list.append(outputs.last_hidden_state.mean(dim=1))
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        embeddings = torch.cat(emb_list, dim=0)
        
        if convert_to_tensor:
            return embeddings.squeeze(0) if single_input else embeddings
        else:
            result = embeddings.cpu().numpy()
            return result.squeeze(0) if single_input else result

if __name__ == "__main__":
    embedder = EmbeddingModel()
    embedding = embedder.get_embedding("What is the capital of France?")
    print(embedding.shape)
