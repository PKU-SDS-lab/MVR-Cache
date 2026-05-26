#!/usr/bin/env python3
import math

def find_factors(n):
    """Find all factors of n."""
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)

def find_dimension_pairs(total_elements):
    """Find reasonable dimension pairs for a matrix."""
    factors = find_factors(total_elements)
    
    print(f"Total elements: {total_elements:,}")
    print(f"Looking for reasonable dimension pairs...")
    
    reasonable_pairs = []
    for i in factors:
        if i > 1000:  # Both dimensions should be > 1000
            j = total_elements // i
            if j > 1000:
                reasonable_pairs.append((i, j))
    
    print(f"\nReasonable dimension pairs (both > 1000):")
    for i, j in reasonable_pairs[:20]:  # Show first 20
        print(f"  {i:,} x {j:,} = {i * j:,}")
    
    return reasonable_pairs

if __name__ == "__main__":
    total_elements = 24065178622
    pairs = find_dimension_pairs(total_elements)
