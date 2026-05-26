import numpy as np
import struct

def read_binary_similarities(file_path):
    """
    Read binary similarity results file produced by the C++ test code.
    
    The binary format is:
    - Header: 2 x uint32_t (NQ, N)
    - Data: NQ x N x float32 (similarities for each query)
    
    Args:
        file_path (str): Path to the binary file
        
    Returns:
        tuple: (NQ, N, similarities_matrix) where:
            - NQ: number of queries
            - N: number of data vectors  
            - similarities_matrix: numpy array of shape (NQ, N) containing similarities
    """
    with open(file_path, 'rb') as f:
        # Read header: NQ and N as uint32_t
        nq_bytes = f.read(4)
        n_bytes = f.read(4)
        
        if len(nq_bytes) != 4 or len(n_bytes) != 4:
            raise ValueError("File too short to contain header")
            
        NQ = struct.unpack('<I', nq_bytes)[0]  # little-endian uint32
        N = struct.unpack('<I', n_bytes)[0]    # little-endian uint32
        
        print(f"Reading binary file: {NQ} queries, {N} data vectors")
        
        # Read all similarity data as float32
        data_bytes = f.read()
        expected_bytes = NQ * N * 4  # 4 bytes per float32
        
        if len(data_bytes) != expected_bytes:
            raise ValueError(f"File size mismatch. Expected {expected_bytes} bytes, got {len(data_bytes)}")
        
        # Convert bytes to numpy array and reshape
        similarities = np.frombuffer(data_bytes, dtype=np.float32).reshape(NQ, N)
        
        return NQ, N, similarities

def read_binary_similarities_chunked(file_path, query_chunk_size=1000):
    """
    Read binary similarity results file in chunks to reduce memory usage.
    
    Args:
        file_path (str): Path to the binary file
        query_chunk_size (int): Number of queries to read at once
        
    Yields:
        tuple: (query_start_idx, query_end_idx, similarities_chunk) for each chunk
    """
    with open(file_path, 'rb') as f:
        # Read header
        nq_bytes = f.read(4)
        n_bytes = f.read(4)
        
        if len(nq_bytes) != 4 or len(n_bytes) != 4:
            raise ValueError("File too short to contain header")
            
        NQ = struct.unpack('<I', nq_bytes)[0]
        N = struct.unpack('<I', n_bytes)[0]
        
        print(f"Reading binary file in chunks: {NQ} queries, {N} data vectors")
        
        # Read data in chunks
        for query_start in range(0, NQ, query_chunk_size):
            query_end = min(query_start + query_chunk_size, NQ)
            chunk_size = query_end - query_start
            
            # Read chunk of similarities
            chunk_bytes = f.read(chunk_size * N * 4)
            if len(chunk_bytes) != chunk_size * N * 4:
                raise ValueError(f"Unexpected end of file at query {query_start}")
            
            similarities_chunk = np.frombuffer(chunk_bytes, dtype=np.float32).reshape(chunk_size, N)
            
            yield query_start, query_end, similarities_chunk

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python read_binary_results.py <binary_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # Read entire file
        NQ, N, similarities = read_binary_similarities(file_path)
        print(f"Successfully read {NQ} queries with {N} similarities each")
        print(f"Similarities shape: {similarities.shape}")
        print(f"First query similarities (first 10): {similarities[0, :10]}")
        
        # Example of chunked reading
        print("\n--- Chunked reading example ---")
        for query_start, query_end, chunk in read_binary_similarities_chunked(file_path, query_chunk_size=100):
            print(f"Read queries {query_start}-{query_end-1}, chunk shape: {chunk.shape}")
            if query_start >= 200:  # Stop after a few chunks for demo
                break
                
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)