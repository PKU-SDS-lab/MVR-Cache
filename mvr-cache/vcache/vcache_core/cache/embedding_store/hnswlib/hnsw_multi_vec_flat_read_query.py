# ==============================================================================
# ### 动态添加项目根目录到系统路径 ###
# 确保 Python 可以找到 index_utils 等自定义模块
import os
import sys

DIM = 128

# 获取当前脚本文件所在的绝对路径
# __file__ 是一个内置变量，代表当前脚本的文件名
current_file_path = os.path.abspath(__file__)

# 从文件路径中获取其所在的目录路径
# 例如：/data/lijunlin/Decompose_retrieval/hnsw_multi_vec_flat_read_query.py -> /data/lijunlin/Decompose_retrieval
project_root_dir = os.path.dirname(current_file_path)

# 将项目根目录添加到 sys.path 的最前面 (使用 insert(0, ...))
# 这可以确保优先搜索我们自己的模块，避免与系统中的同名模块冲突
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
    print(f"[INFO] Project root added to Python path: {project_root_dir}")
# ==============================================================================

import os
from transformers import CLIPModel, AutoProcessor
import torch
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

import time
import warnings
# from clustering import *
# from text_utils import *
# from LLM4split.prompt_utils import *
# from raptor.raptor_embeddings import *
import pickle
import pandas as pd
# from hnsw import HNSW
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import List, Tuple, Dict
import hashlib

# Import the MaxSim index implementations
# Ensure simple_maxsim is available
try:
    from index_utils.simple_maxsim import HybridMaxSimIndex, SimpleMaxSimIndex
except ImportError as e:
    print("\n" + "="*80)
    print("--- DETAILED IMPORT ERROR ---")
    print(f"An ImportError occurred. The actual error is:")
    print(f"\n    ERROR: {e}\n")
    print("This means either the module was not found, OR a module it tried to import was not found.")
    print("="*80 + "\n")
    
    # 为了进一步调试，打印出 Python 当前的搜索路径
    import sys
    print("Python is searching for modules in these paths:")
    for p in sys.path:
        print(f"  - {p}")
    print("\nCheck if your project root '/data/lijunlin/Decompose_retrieval' is listed above.")
    exit(1)

sbert_path = '/data/lijunlin/models/msmarco-roberta-base-v2'

# Add new class for MaxSim indexing with method selection
class MaxSimFAISSIndexer:
    """
    Complete MaxSim indexer with FAISS HNSW, caching support, and method selection
    """
    
    def __init__(self, model_path="sentence-transformers/msmarco-roberta-base-v2", 
                 cache_dir="./maxsim_cache", M=16, ef_construction=200, ef_search=50,
                 index_method="hybrid", index_dir="./index", device='cuda'):
        """
        Initialize the MaxSim indexer
        
        Args:
            model_path: Path to the sentence transformer model
            cache_dir: Directory to store cached indices (deprecated, kept for compatibility)
            M: HNSW parameter - number of connections
            ef_construction: HNSW parameter for construction
            ef_search: HNSW parameter for search
            index_method: "simple" for SimpleMaxSimIndex, "hybrid" for HybridMaxSimIndex
            index_dir: Directory to store indices
            device: Device to run on ('cuda' or 'cpu')
        """
        print(f"Initializing MaxSim FAISS Indexer with method: {index_method}")
        print(f"HNSW Parameters: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Index method selection
        self.index_method = index_method
        
        # HNSW parameters
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.d = DIM  # SBERT embedding dimension
        
        # Index directory
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        # Cache directory (kept for backward compatibility)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def set_hnsw_parameters(self, M=None, ef_construction=None, ef_search=None):
        """
        Update HNSW parameters
        
        Args:
            M: HNSW parameter - number of connections
            ef_construction: HNSW parameter for construction
            ef_search: HNSW parameter for search
        """
        if M is not None:
            self.M = M
            # print(f"Updated M to {M}") # Removed for cleaner logs during batch runs
        if ef_construction is not None:
            self.ef_construction = ef_construction
            # print(f"Updated ef_construction to {ef_construction}")
        if ef_search is not None:
            self.ef_search = ef_search
            # print(f"Updated ef_search to {ef_search}")
    
    def get_hnsw_parameters(self):
        """
        Get current HNSW parameters
        
        Returns:
            dict: Dictionary containing current HNSW parameters
        """
        return {
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search
        }
        
    def encode_token_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """
        Encode texts into token embeddings (keeping as tensors)
        
        Args:
            texts: List of text strings
            
        Returns:
            embeddings: List of tensors, each of shape [N, D]
        """
        embeddings = []
        
        print(f"Encoding {len(texts)} texts...")
        for text in tqdm(texts, desc="Encoding"):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs).last_hidden_state[0]
            
            # Remove [CLS] and [SEP] tokens
            token_emb = outputs[1:-1].cpu()
            embeddings.append(token_emb)
        
        return embeddings
    
    def _compute_corpus_hash(self, corpus_embeddings: List[torch.Tensor]) -> str:
        """
        Compute a hash for the corpus to use as cache key
        """
        # Create a hash from the shapes and some sample values
        hash_data = []
        for i, emb in enumerate(corpus_embeddings[:min(10, len(corpus_embeddings))]):
            hash_data.append((i, emb.shape))
            if emb.shape[0] > 0:
                hash_data.append(emb[0, :5].numpy().tobytes())
        
        # Add index method to hash
        hash_data.append(f"method_{self.index_method}")
        
        hash_str = str(hash_data).encode()
        return hashlib.md5(hash_str).hexdigest()[:16]
    
    def _get_index_filename(self):
        """
        Generate index filename based on HNSW parameters
        
        Returns:
            str: Index filename
        """
        index_name = f"improved_openai_hnsw_M{self.M}_efc{self.ef_construction}_efs{self.ef_search}_{self.index_method}"
        return index_name
    
    def _get_index_path(self):
        """
        Get the full path for the index file
        
        Returns:
            str: Full path to index file
        """
        index_name = self._get_index_filename()
        index_path = os.path.join(self.index_dir, f"{index_name}.pkl")
        return index_path
    
    def build_index(self, 
                      corpus_embeddings: List[torch.Tensor], 
                      index_path: str = None,
                      rebuild: bool = False) -> Tuple[object, float]:
        """
        Build index for the corpus with caching (only for hybrid method)
        For simple method, just store the embeddings without building an index
        
        Args:
            corpus_embeddings: List of document embeddings (tensors)
            index_path: Custom path to save/load the index. If None, uses default path based on parameters
            rebuild: Force rebuilding even if cache exists
            
        Returns:
            index: Built index (SimpleMaxSimIndex or HybridMaxSimIndex) or corpus embeddings for simple method
            build_time: Time taken to build (0 if loaded from cache or using simple method)
        """
        # For simple method, we don't need to build an index
        if self.index_method == "simple":
            print(f"Using simple method - no index building required")
            # Normalize all corpus embeddings using F.normalize
            normalized_corpus = []
            for emb in corpus_embeddings:
                emb = emb.to(self.device)
                emb_normalized = F.normalize(emb, p=2, dim=1)
                normalized_corpus.append(emb_normalized.cpu())
            return normalized_corpus, 0.0
        
        # For hybrid method, proceed with index building
        if index_path is None:
            index_path = self._get_index_path()
        else:
            # Ensure the directory exists
            index_dir = os.path.dirname(index_path)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
        
        # Try to load from cache if it exists and conditions are met
        if not rebuild and os.path.exists(index_path):
            print(f"Loading index from: {index_path}")
            try:
                with open(index_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Verify the cached index matches current parameters
                cached_params = cached_data.get('hnsw_parameters', {})
                current_params = self.get_hnsw_parameters()
                
                if (cached_params.get('M') == current_params['M'] and 
                    cached_params.get('ef_construction') == current_params['ef_construction'] and
                    cached_data.get('method') == self.index_method and
                    cached_data.get('n_docs') == len(corpus_embeddings)):
                    
                    print(f"Successfully loaded cached index")
                    print(f"Index info: {cached_data['n_docs']} documents, "
                          f"built in {cached_data.get('build_time', 0):.2f}s")
                    
                    # Update ef_search if different
                    if hasattr(cached_data['index'], 'index') and hasattr(cached_data['index'].index, 'efSearch'):
                        cached_data['index'].index.efSearch = self.ef_search
                        print(f"Updated ef_search to {self.ef_search}")
                    
                    return cached_data['index'], 0.0
                else:
                    print(f"Cached index parameters don't match current settings, rebuilding...")
            except Exception as e:
                print(f"Error loading cached index: {e}")
                print(f"Will rebuild index...")
        
        # Build new index if cache doesn't exist or rebuild is True
        print(f"Building new {self.index_method} index...")
        # print(f"Index will be saved to: {index_path}")
        start_time = time.time()
        
        # 1. Create the index object WITHOUT ef_construction
        index = HybridMaxSimIndex(self.d, self.M)

        # 2. Set efConstruction as a property on the underlying HNSW index BEFORE building
        if hasattr(index, 'index') and hasattr(index.index, 'efConstruction'):
            index.index.efConstruction = self.ef_construction
        
        # Convert PyTorch tensors to numpy arrays and extract token counts
        doc_embeddings_np = []
        token_counts = []
        
        for emb_tensor in corpus_embeddings:
            # Normalize using F.normalize
            emb_tensor = emb_tensor.to(self.device)
            emb_normalized = F.normalize(emb_tensor, p=2, dim=1)
            emb_np = emb_normalized.cpu().numpy()
            doc_embeddings_np.append(emb_np)
            token_counts.append(len(emb_np))  # Number of tokens in this document
        
        # Add all documents to the index
        index.add_with_token_counts(doc_embeddings_np, token_counts)
        
        # Set ef_search after building
        if hasattr(index, 'index') and hasattr(index.index, 'efSearch'):
            index.index.efSearch = self.ef_search
        
        build_time = time.time() - start_time
        print(f"Built {self.index_method} index in {build_time:.2f} seconds")
        
        # Commented out saving index to avoid clutter during batch tests
        # print(f"Saving index to: {index_path}")
        # with open(index_path, 'wb') as f:
        #     pickle.dump({
        #         'index': index,
        #         'build_time': build_time,
        #         'n_docs': len(corpus_embeddings),
        #         'timestamp': time.time(),
        #         'method': self.index_method,
        #         'hnsw_parameters': self.get_hnsw_parameters()
        #     }, f)
        
        return index, build_time
    
    def compute_maxsim_simple(self, query_emb, doc_emb):
        """
        Compute MaxSim between query and document embeddings (using average)
        
        Args:
            query_emb: Query embedding tensor of shape [n_query_tokens, d]
            doc_emb: Document embedding tensor of shape [n_doc_tokens, d]
            
        Returns:
            score: MaxSim score (average of max similarities)
        """
        # Ensure both tensors are on the same device
        query_emb = query_emb.to(self.device)
        doc_emb = doc_emb.to(self.device)
        
        # Normalize embeddings using F.normalize
        query_norm = F.normalize(query_emb, p=2, dim=1)
        doc_norm = F.normalize(doc_emb, p=2, dim=1)
        
        # Compute cosine similarity matrix between all query and doc tokens
        sim_matrix = torch.matmul(query_norm, doc_norm.T)
        
        # For each query token, find max similarity with any doc token
        max_sims, _ = torch.max(sim_matrix, dim=1)
        
        # Average the max similarities (instead of sum)
        return max_sims.mean().item()
    
    def search(self, 
               query_embeddings: List[torch.Tensor],
               index_or_corpus: object,
               k: int = 10,
               save_top_k: int = 100) -> Tuple[np.ndarray, np.ndarray, List[Dict], float]:
        """
        Search the index or compute MaxSim directly for simple method
        
        Args:
            query_embeddings: List of query embeddings (tensors)
            index_or_corpus: The index (HybridMaxSimIndex) or corpus embeddings list for simple method
            k: Number of results per query
            save_top_k: Number of top results to save for each query
            
        Returns:
            distances: Distance matrix of shape (n_queries, k)
            indices: Index matrix of shape (n_queries, k)
            full_results: List of dicts with top-save_top_k indices and scores
            search_time: Total search time
        """
        print(f"\nPerforming retrieval for {len(query_embeddings)} queries using {self.index_method} method...")
        
        start_time = time.time()
        full_results = []  # Store full top-save_top_k results
        
        if self.index_method == "simple":
            # Direct MaxSim computation without index
            corpus_embeddings = index_or_corpus  # It's actually the corpus embeddings list
            n_queries = len(query_embeddings)
            n_docs = len(corpus_embeddings)
            
            # Initialize result arrays
            all_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
            
            # Compute MaxSim for each query-document pair
            for q_idx, query_emb in enumerate(tqdm(query_embeddings, desc="Computing MaxSim")):
                for d_idx, doc_emb in enumerate(corpus_embeddings):
                    score = self.compute_maxsim_simple(query_emb, doc_emb)
                    all_scores[q_idx, d_idx] = score
            
            # Get top-save_top_k results for each query
            distances = np.full((n_queries, k), -np.inf, dtype=np.float32)
            indices = np.full((n_queries, k), -1, dtype=np.int64)
            
            for q_idx in range(n_queries):
                # Get indices of top-save_top_k scores
                top_save_k_indices = np.argpartition(-all_scores[q_idx], min(save_top_k, n_docs-1))[:save_top_k]
                top_save_k_indices = top_save_k_indices[np.argsort(-all_scores[q_idx][top_save_k_indices])]
                
                # Save full top-save_top_k results
                full_results.append({
                    'indices': top_save_k_indices.tolist(),
                    'scores': all_scores[q_idx][top_save_k_indices].tolist()
                })
                
                # Fill in top-k results
                for i in range(min(k, len(top_save_k_indices))):
                    idx = top_save_k_indices[i]
                    distances[q_idx, i] = -all_scores[q_idx, idx]  # Negative for consistency
                    indices[q_idx, i] = idx
            
        else:
            # For hybrid method, use the existing index-based search
            index = index_or_corpus  # It's actually the HybridMaxSimIndex
            
            # Convert query tensors to numpy arrays for FAISS processing
            query_embeddings_np = []
            query_token_counts = []
            
            for emb_tensor in query_embeddings:
                # Normalize using F.normalize
                emb_tensor = emb_tensor.to(self.device)
                emb_normalized = F.normalize(emb_tensor, p=2, dim=1)
                emb_np = emb_normalized.cpu().numpy()
                query_embeddings_np.append(emb_np)
                query_token_counts.append(len(emb_np))
            
            # Perform MaxSim-based search using the HNSW index
            # Search for save_top_k results
            distances_full, indices_full = index.search_with_token_counts(
                query_embeddings_np, query_token_counts, save_top_k
            )
            
            # Convert from sum to average by dividing by query token counts
            for q_idx in range(len(query_embeddings)):
                if query_token_counts[q_idx] > 0:
                    distances_full[q_idx] = distances_full[q_idx] / query_token_counts[q_idx]
                
                # Save full top-save_top_k results
                full_results.append({
                    'indices': indices_full[q_idx].tolist(),
                    'scores': (-distances_full[q_idx]).tolist()  # Convert to positive scores
                })
            
            # Extract top-k results
            distances = distances_full[:, :k]
            indices = indices_full[:, :k]
        
        search_time = time.time() - start_time
        print(f"Retrieval completed in {search_time:.4f} seconds")
        
        return distances, indices, full_results, search_time


def save_retrieval_results(full_results, output_path):
    """
    Save retrieval results to disk in pickle format.
    
    Args:
        full_results: List of dicts with 'indices' and 'scores' for each query
        output_path: Path to save the results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save using pickle for efficiency
    with open(output_path, 'wb') as f:
        pickle.dump(full_results, f)
    
    print(f"Retrieval results (PKL) saved to: {output_path}")
    
    # Also save metadata
    metadata = {
        'num_queries': len(full_results),
        'top_k_saved': len(full_results[0]['indices']) if full_results else 0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = output_path.replace('.pkl', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

# ----------------- 新增功能：保存为 TXT 文件 -----------------
def save_results_to_txt(full_results, output_path):
    """
    Save retrieval results to a txt file in the specified format.
    Format: query_id<tab>passage_id<tab>rank<tab>score
    
    Args:
        full_results: List of dicts with 'indices' and 'scores' for each query.
        output_path: Path to the output .txt file.
    """
    print(f"Saving results in TXT format to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        # Iterate through each query's results with its query_id
        for query_id, result in enumerate(full_results):
            passage_ids = result['indices']
            scores = result['scores']
            # Iterate through passages and scores, getting the rank (starting from 1)
            for rank, (passage_id, score) in enumerate(zip(passage_ids, scores), 1):
                f.write(f"{query_id}\t{passage_id}\t{rank}\t{score}\n")
    print("TXT results saved successfully.")
# -------------------------------------------------------------

def read_data(dataset):
    """
    加载由四个 .npy 文件定义的 query 和 corpus embedding。
    现在支持二维连续的 query_embedding 输入。
    """
    print("="*50)
    print(f"Loading data for dataset: {dataset}...")
    
    dataset_name = dataset
    # 定义文件路径
    data_dir = f"/data/sigmod2025/{dataset_name}"
    if dataset_name == "clerc":
        query_emb_path = os.path.join(data_dir, "full_multi_embeddings_clerc_query.npy")
        query_len_path = os.path.join(data_dir, "full_multi_chunk_num_clerc_query.npy")
        corpus_emb_path = os.path.join(data_dir, "full_multi_embeddings_clerc.npy")
        corpus_len_path = os.path.join(data_dir, "full_multi_chunk_num_clerc.npy")
    elif dataset_name == "clerc_large":
        query_emb_path = os.path.join(data_dir, "full_multi_embeddings_clerc_large_query.npy")
        query_len_path = os.path.join(data_dir, "full_multi_chunk_num_clerc_large_query.npy")
        corpus_emb_path = os.path.join(data_dir, "full_multi_embeddings_clerc_large.npy")
        corpus_len_path = os.path.join(data_dir, "full_multi_chunk_num_clerc_large.npy")
    elif dataset_name == "clerc_small":
        query_emb_path = os.path.join(data_dir, "full_multi_embeddings_clerc_query.npy")
        query_len_path = os.path.join(data_dir, "full_multi_chunk_num_clerc_query.npy")
        corpus_emb_path = os.path.join(data_dir, "full_embeddings.npy")
        corpus_len_path = os.path.join(data_dir, "full_multi_chunk_num_clerc.npy")
    elif dataset_name == "clerc_128":
        data_dir = f"/data/ali/clerc-128-small-multi"
        query_emb_path = os.path.join(data_dir, "full_multi_embeddings_clerc_128_small_query.npy")
        query_len_path = os.path.join(data_dir, "full_multi_chunk_num_clerc_128_small_query.npy")
        corpus_emb_path = os.path.join(data_dir, "full_multi_embeddings_clerc_128_small.npy")
        corpus_len_path = os.path.join(data_dir, "full_multi_chunk_num_clerc_128_small.npy")
        

    # 加载 NumPy 数组
    query_embeddings = np.load(query_emb_path, mmap_mode='r')
    all_corpus_embeddings = np.load(corpus_emb_path, mmap_mode='r')
    
    query_lengths = np.load(query_len_path).astype(np.int64) # 确保是整数
    doc_lengths = np.load(corpus_len_path).astype(np.int64)

    # --- 输入数据检查 ---
    print("\n--- Input Data Verification ---")
    print(f"Query Embeddings Shape : {query_embeddings.shape}, Type: {query_embeddings.dtype}")
    print(f"Query Lengths Shape    : {query_lengths.shape}, Type: {query_lengths.dtype}")
    print(f"Corpus Embeddings Shape: {all_corpus_embeddings.shape}, Type: {all_corpus_embeddings.dtype}")
    print(f"Corpus Lengths Shape   : {doc_lengths.shape}, Type: {doc_lengths.dtype}")
    
    # --- 核心改动 1: 增加对二维 Query 数据的完整性检查 ---
    # 检查查询向量总数和长度数组总和是否匹配
    # 只有当 query_embeddings 是二维时才进行此检查
    if len(query_embeddings.shape) == 2:
        print("\n--- Additional Query Data Verification (for 2D input) ---")
        total_query_vectors_calculated = np.sum(query_lengths)
        total_query_vectors_actual = query_embeddings.shape[0]
        print(f"Sum of query lengths: {total_query_vectors_calculated}")
        print(f"Total vectors in query_embedding.npy: {total_query_vectors_actual}")
        assert total_query_vectors_calculated == total_query_vectors_actual, \
            "Mismatch between query lengths and total query vectors in the 2D embedding file!"
        print("Query data consistency check passed for 2D input.")
        print("-------------------------------------------------------\n")
    
    # 检查语料库长度和总 token 数是否匹配
    total_tokens_calculated = np.sum(doc_lengths)
    total_tokens_actual = all_corpus_embeddings.shape[0]
    print(f"Sum of doc lengths: {total_tokens_calculated}")
    print(f"Total tokens in embedding.npy: {total_tokens_actual}")
    assert total_tokens_calculated == total_tokens_actual, "Mismatch between doc lengths and total tokens!"
    print("Corpus data consistency check passed.")
    print("---------------------------------\n")

    # --- 核心改动 2: 修改重建 query_list 的逻辑 ---
    print("Reconstructing query list from 2D tensor...")
    query_list = []
    start_idx = 0 # 初始化起始指针
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        # 遍历每个查询的向量长度
        for length in tqdm(query_lengths, desc="Processing Queries"):
            end_idx = start_idx + length # 计算结束指针
            # 从大的二维数组中切片，获取当前查询的所有向量
            query_vectors = query_embeddings[start_idx:end_idx, :]
            query_list.append(torch.from_numpy(query_vectors).float())
            start_idx = end_idx # 更新起始指针以备下一次迭代

    # 重建 corpus_list 的逻辑保持不变
    print("Reconstructing corpus list (this may take a moment)...")
    corpus_list = []
    start_idx = 0
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for length in tqdm(doc_lengths, desc="Processing Corpus"):
            end_idx = start_idx + length
            doc_vectors = all_corpus_embeddings[start_idx:end_idx, :]
            corpus_list.append(torch.from_numpy(doc_vectors).float())
            start_idx = end_idx

    # --- 最终输出格式检查 ---
    print("\n--- Final Data Structure Verification ---")
    if corpus_list:
      for i in range(min(3, len(corpus_list))):
          print(f"corpus_list[{i}] shape: {corpus_list[i].shape}, type: {corpus_list[i].dtype}")
    if query_list:
      for i in range(min(3, len(query_list))):
          print(f"query_list[{i}] shape: {query_list[i].shape}, type: {query_list[i].dtype}")

    print(f"\nTotal number of items in corpus_list: {len(corpus_list)}")
    print(f"Total number of items in query_list: {len(query_list)}")
    print("="*50 + "\n")
    
    return corpus_list, query_list


def save_summary_to_txt(summary_path, num_docs, num_queries, index_method, hnsw_params,
                        device, build_time, search_time, k_sim, accuracy, mrr, 
                        results_paths):
    """
    将运行摘要和评估指标保存到文本文件中。
    """
    avg_retrieval_ms = (search_time / num_queries) * 1000 if num_queries > 0 else 0

    summary_content = f"""
==================================================
                 Run Summary
==================================================
Timestamp:           {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Data:
--------------------------------------------------
Total documents:     {num_docs}
Total queries:       {num_queries}

Configuration:
--------------------------------------------------
Index method:        {index_method}
Index parameters:    {hnsw_params}
Device:              {device}

Performance:
--------------------------------------------------
Index build time:    {build_time:.6f} seconds
Retrieval time:      {search_time:.6f} seconds
Avg retrieval time:  {avg_retrieval_ms:.2f} ms/query

Evaluation Metrics:
--------------------------------------------------
Accuracy (Hit@{k_sim}):    {accuracy:.4f}
MRR@{k_sim}:               {mrr:.4f}

Output Files:
--------------------------------------------------
PKL results:         {results_paths['pkl']}
TXT results:         {results_paths['txt']}
==================================================
"""
    # 移除开头的空行
    summary_content = summary_content.strip()

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"Run summary saved to: {summary_path}")
    

if __name__ == "__main__":

    # --- 1. 配置批量测试参数 ---
    # 定义您想测试的 HNSW 参数组合
    param_sets = [
        {'M': 16, 'ef_construction': 400, 'ef_search': 100},
        {'M': 32, 'ef_construction': 200, 'ef_search': 50},
        {'M': 8,  'ef_construction': 100, 'ef_search': 30},
        {'M': 16, 'ef_construction': 200, 'ef_search': 100},
        {'M': 4, 'ef_construction': 400, 'ef_search': 100},
        {'M': 8, 'ef_construction': 400, 'ef_search': 50},
        
        {'M': 32, 'ef_construction': 400, 'ef_search': 100},
        {'M': 4, 'ef_construction': 100, 'ef_search': 50},
    ]
    dataset = "clerc_small" 
    
    # clerc
    # clerc_large
    # clerc_small
    # clerc_128
    
    # 为每组参数设置运行次数 (用于测试稳定性)
    num_runs = 1  # 如果不需要重复测试, 设置为 1 即可

    # --- 2. 初始化 ---
    # 加载一次数据即可
    corpus_emb_list, query_emb_list_full = read_data(dataset)
    num_original_queries = len(query_emb_list_full) # 保存原始查询数量用于评估

    # ##########################################################################
    # ### ↓↓↓ 新增功能: 如果查询数量超过1000，则仅使用前1000条进行测试 ↓↓↓ ###
    # ##########################################################################
    max_queries_to_test = 1000
    if num_original_queries > max_queries_to_test:
        print(f"\n[INFO] Total queries loaded: {num_original_queries}. Truncating to the first {max_queries_to_test} for this retrieval test.")
        query_emb_list = query_emb_list_full[:max_queries_to_test]
    else:
        query_emb_list = query_emb_list_full
        print(f"\n[INFO] Total queries loaded: {num_original_queries}. Using all of them for retrieval test.")
    # ##########################################################################
    # ### ↑↑↑ 功能添加结束 ↑↑↑ #################################################
    # ##########################################################################

    k_sim = 10
    save_top_k = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    maxsim_method = "hybrid" # 假设测试中始终使用 hybrid 方法
    
    
    # 用于存储所有运行结果的列表, 以便最后生成总览报告
    all_runs_summary_data = []

    # --- 3. 循环执行测试 ---
    # 外层循环: 遍历参数组合
    for i, params in enumerate(param_sets):
        # 内层循环: 重复运行
        for run_num in range(1, num_runs + 1):
            
            # 提取当前运行的参数
            M = params['M']
            ef_construction = params['ef_construction']
            ef_search = params['ef_search']

            print("\n" + "#"*90)
            print(f"### STARTING TEST: Set {i+1}/{len(param_sets)}, Run {run_num}/{num_runs} ###")
            print(f"### Parameters: M={M}, ef_construction={ef_construction}, ef_search={ef_search} ###")
            print("#"*90 + "\n")
            
            # 初始化索引器
            indexer = MaxSimFAISSIndexer(
                model_path=sbert_path,
                index_method=maxsim_method,
                M=M,
                ef_construction=ef_construction,
                ef_search=ef_search,
                device=device
            )
            
            print(f"\nCurrent HNSW parameters: {indexer.get_hnsw_parameters()}")
            print(f"Device: {device}")
            
            # 构建索引
            index_name = f"{dataset}_hnsw_M{M}_efc{ef_construction}"
            index_path = f"./index/{index_name}.pkl"
            index, build_time = indexer.build_index(
                corpus_emb_list,
                index_path=index_path,
                rebuild=True # 强制为每个测试重建索引
            )
            
            # 执行检索
            print("\nStep: Performing retrieval...")
            distances, indices, full_results, search_time = indexer.search(
                query_emb_list, # 使用可能被截断的查询列表
                index, 
                k=k_sim,
                save_top_k=save_top_k
            )
            
            # 为本次运行定义独一无二的输出路径
            output_dir = "./results"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 文件名包含参数和运行次数, 确保唯一性
            base_filename = f"{dataset}_HNSW_M{M}_efc{ef_construction}_efs{ef_search}_run{run_num}"
            
            pkl_output_path = os.path.join(output_dir, f"{base_filename}.pkl")
            txt_output_path = os.path.join(output_dir, f"{base_filename}.txt")
            summary_output_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
            
            # 保存结果 (PKL 和 TXT)
            save_retrieval_results(full_results, pkl_output_path)
            save_results_to_txt(full_results, txt_output_path)
            
            # 评估结果
            right = 0
            mrr_sum = 0
            # 循环次数基于截断后的查询列表长度
            for q_idx in range(len(query_emb_list)):
                # #############################################################
                # ### ↓↓↓ 核心修改: 使用原始查询总数计算正确的ID, 保证评估准确性 ↓↓↓ ###
                # #############################################################
                right_passage_id = q_idx + len(corpus_emb_list) - num_original_queries
                # #############################################################

                # 直接从 `indices` 数组中查找
                for rank, doc_idx in enumerate(indices[q_idx]):
                    if doc_idx == right_passage_id:
                        right += 1
                        mrr_sum += 1.0 / (rank + 1)
                        break
            
            # 分母使用截断后的查询列表长度
            accuracy = right / len(query_emb_list) if len(query_emb_list) > 0 else 0
            mrr = mrr_sum / len(query_emb_list) if len(query_emb_list) > 0 else 0

            # 保存单次运行的摘要文件
            save_summary_to_txt(
                summary_path=summary_output_path,
                num_docs=len(corpus_emb_list),
                num_queries=len(query_emb_list), # 报告截断后的数量
                index_method=maxsim_method,
                hnsw_params=indexer.get_hnsw_parameters(),
                device=device,
                build_time=build_time,
                search_time=search_time,
                k_sim=k_sim,
                accuracy=accuracy,
                mrr=mrr,
                results_paths={'pkl': pkl_output_path, 'txt': txt_output_path}
            )

            # 将本次运行的结果存入总览列表
            run_summary = {
                'M': M,
                'ef_construction': ef_construction,
                'ef_search': ef_search,
                'run': run_num,
                'build_time_s': f"{build_time:.4f}",
                'search_time_s': f"{search_time:.4f}",
                f'accuracy_at_{k_sim}': f"{accuracy:.4f}",
                f'mrr_at_{k_sim}': f"{mrr:.4f}",
                'avg_retrieval_ms_per_query': f"{(search_time / len(query_emb_list)) * 1000:.2f}"
            }
            all_runs_summary_data.append(run_summary)
            
            print(f"\n### FINISHED TEST: Set {i+1}, Run {run_num}. Results saved. ###")
            print("#"*90)

    # --- 4. 输出并保存最终的汇总报告 ---
    print("\n" + "*"*90)
    print(" " * 32 + "FINAL BATCH TEST SUMMARY")
    print("*"*90 + "\n")

    if all_runs_summary_data:
        # 使用 pandas 创建 DataFrame 以便清晰地展示和保存
        summary_df = pd.DataFrame(all_runs_summary_data)
        
        # 在控制台打印总览表格
        print("--- All Runs Summary ---")
        print(summary_df.to_string())
        
        # 将总览表格保存为 CSV 文件
        final_summary_filename = os.path.join("./results", f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        summary_df.to_csv(final_summary_filename, index=False)
        print(f"\nFinal consolidated summary saved to: {final_summary_filename}")
    else:
        print("No tests were run or no data was collected.")