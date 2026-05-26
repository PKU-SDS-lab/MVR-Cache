#!/usr/bin/env python3
import numpy as np
import struct
import math

def diagnose_binary_file(filename):
    """Diagnose the binary file to find the correct dimensions."""
    with open(filename, 'rb') as f:
        # Read header
        nq = struct.unpack('<I', f.read(4))[0]  # little-endian uint32
        n = struct.unpack('<I', f.read(4))[0]   # little-endian uint32
        
        print(f"Header values:")
        print(f"  NQ (queries): {nq}")
        print(f"  N (data points): {n}")
        print(f"  Expected data size: {nq * n * 4} bytes")
        
        # Get actual file size
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        f.seek(8)  # Seek past header
        
        actual_data_size = file_size - 8
        actual_float32_count = actual_data_size // 4
        
        print(f"\nActual file analysis:")
        print(f"  Total file size: {file_size:,} bytes")
        print(f"  Data size (after header): {actual_data_size:,} bytes")
        print(f"  Number of float32 values: {actual_float32_count:,}")
        
        # Try to find factors that could be the correct dimensions
        print(f"\nLooking for factors of {actual_float32_count:,}:")
        
        # Find all factors
        factors = []
        for i in range(1, int(math.sqrt(actual_float32_count)) + 1):
            if actual_float32_count % i == 0:
                factors.append(i)
                if i != actual_float32_count // i:
                    factors.append(actual_float32_count // i)
        
        factors.sort()
        
        # Show some reasonable factor pairs
        print(f"  Some possible dimension pairs:")
        count = 0
        for i in factors:
            if i > 1000 and actual_float32_count // i > 1000:  # Both dimensions > 1000
                print(f"    {i} x {actual_float32_count // i}")
                count += 1
                if count >= 10:  # Limit output
                    break
        
        # Check if the original dimensions make sense
        print(f"\nValidation:")
        print(f"  Original NQ * N = {nq} * {n} = {nq * n:,}")
        print(f"  Actual float32 count = {actual_float32_count:,}")
        print(f"  Match: {nq * n == actual_float32_count}")
        
        if nq * n != actual_float32_count:
            print(f"  Mismatch! The header values don't match the actual data size.")
            print(f"  This suggests the file was written with different dimensions than expected.")

if __name__ == "__main__":
    filename = "/data1/ali/similarities/fullopenaidecompose_hnsw_similarities.bin"
    diagnose_binary_file(filename)
