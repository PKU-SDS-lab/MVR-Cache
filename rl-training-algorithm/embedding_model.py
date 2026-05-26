import os

# Allow users to choose a mirror explicitly, but do not force a machine-specific default.
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
                local_models = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "BAAI__bge-base-en-v1.5"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "bge-base-en"),
                ]
                model_name = next(
                    (os.path.normpath(path) for path in local_models if os.path.isdir(path)),
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
            # Use HuggingFace hub (model_name is a valid repo ID)
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
        
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
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

        # Decide the device to run on.
        # If caller didn't specify, use the model's current device.
        run_device = device
        if run_device is not None:
            self.model.to(run_device)
        else:
            run_device = next(self.model.parameters()).device

        with torch.no_grad():
            for start_idx in range(0, batch_size_total, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size_total)
                texts_chunk = texts[start_idx:end_idx]
                inputs = self.tokenizer(texts_chunk, return_tensors="pt", padding=True, truncation=True)
                # Always move inputs to the same device as the model to avoid CPU/GPU mismatch.
                inputs = {k: v.to(run_device) for k, v in inputs.items()}

                if run_device is not None and str(run_device).startswith("cuda") and torch.cuda.is_available():
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
