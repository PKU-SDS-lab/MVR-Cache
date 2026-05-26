import numpy as np
import struct
from typing import Tuple, Optional
import os

class SimilarityReader:
    """
    A class to read and work with binary similarity files created by the C++ code.
    
    The binary format is:
    - 4 bytes: NQ (number of queries) as uint32
    - 4 bytes: N (number of data points) as uint32  
    - NQ * N * 4 bytes: similarity values as float32
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        self.similarities = None
        self.nq = None
        self.n = None
        self._load()
    
    def _load(self):
        """Load the binary file into memory."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        
        with open(self.filename, 'rb') as f:
            # Read header
            self.nq = struct.unpack('<I', f.read(4))[0]  # little-endian uint32
            self.n = struct.unpack('<I', f.read(4))[0]   # little-endian uint32
            
            # Read all similarities as a flat array
            similarities_flat = np.frombuffer(f.read(), dtype=np.float32)
            
            # Reshape to (NQ, N) matrix
            self.similarities = similarities_flat.reshape(self.nq, self.n)
    
    def get_similarities(self) -> np.ndarray:
        """Get the full similarity matrix."""
        return self.similarities
    
    def get_query_similarities(self, query_id: int) -> np.ndarray:
        """Get similarities for a specific query."""
        if query_id >= self.nq:
            raise IndexError(f"Query ID {query_id} out of range (max: {self.nq-1})")
        return self.similarities[query_id, :]
    
    def get_top_k(self, query_id: int, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k most similar data points for a query.
        
        Returns:
            Tuple of (indices, similarities) sorted by similarity (descending)
        """
        query_sims = self.get_query_similarities(query_id)
        top_k_indices = np.argsort(query_sims)[-k:][::-1]  # top k, descending order
        top_k_similarities = query_sims[top_k_indices]
        return top_k_indices, top_k_similarities
    
    def get_stats(self) -> dict:
        """Get statistics about the similarity matrix."""
        return {
            'shape': self.similarities.shape,
            'min': float(self.similarities.min()),
            'max': float(self.similarities.max()),
            'mean': float(self.similarities.mean()),
            'std': float(self.similarities.std()),
            'file_size_bytes': os.path.getsize(self.filename)
        }
    
    def save_as_csv(self, output_file: str):
        """Save the similarities as a CSV file (for compatibility)."""
        import pandas as pd
        
        # Create DataFrame
        data = []
        for q_id in range(self.nq):
            for d_id in range(self.n):
                data.append({
                    'query_id': q_id,
                    'data_id': d_id,
                    'similarity': self.similarities[q_id, d_id]
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Saved as CSV: {output_file}")
    
    def save_as_npy(self, output_file: str):
        """Save the similarities as a NumPy .npy file."""
        np.save(output_file, self.similarities)
        print(f"Saved as NumPy array: {output_file}")
    
    def __repr__(self):
        return f"SimilarityReader(filename='{self.filename}', shape={self.similarities.shape})"

# Example usage
if __name__ == "__main__":
    # Example usage
    binary_file = "/home/ali/hnswlib/results/openai400_hnsw_similarities.bin"
    
    if os.path.exists(binary_file):
        # Load the similarities
        reader = SimilarityReader(binary_file)
        print(f"Loaded: {reader}")
        
        # Get statistics
        stats = reader.get_stats()
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Get top-5 for first query
        if reader.nq > 0:
            top_indices, top_sims = reader.get_top_k(0, k=5)
            print(f"\nTop-5 similarities for query 0:")
            for i, (idx, sim) in enumerate(zip(top_indices, top_sims)):
                print(f"  {i+1}. data_id={idx}, similarity={sim:.6f}")
        
        # Show file size comparison
        csv_file = binary_file.replace('.bin', '.csv')
        if os.path.exists(csv_file):
            csv_size = os.path.getsize(csv_file)
            binary_size = stats['file_size_bytes']
            print(f"\nFile size comparison:")
            print(f"  Binary: {binary_size:,} bytes ({binary_size/1024/1024:.2f} MB)")
            print(f"  CSV: {csv_size:,} bytes ({csv_size/1024/1024:.2f} MB)")
            print(f"  Space savings: {((csv_size - binary_size) / csv_size * 100):.1f}%")
    else:
        print(f"Binary file not found: {binary_file}")
        print("Please run the C++ code first to generate the binary file.")
