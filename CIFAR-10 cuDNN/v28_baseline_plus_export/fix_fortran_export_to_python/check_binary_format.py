"""
Check if there's a Fortran base-1 indexing issue.

Fortran prints: conv1_out_host(1, 1, 1, 1) through conv1_out_host(5, 1, 1, 1)
This is W=1-5, H=1, C=1, N=1 in Fortran (1-indexed)

But maybe the ACTUAL data at those indices doesn't match what we think?

Let's also check if maybe the weights file has extra bytes (header) or is missing bytes.
"""

import numpy as np
from pathlib import Path

model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_file = model_dir / "conv1_weights.bin"

# Check file size
file_size = conv1_w_file.stat().st_size
expected_size = 32 * 3 * 3 * 3 * 4  # 4 bytes per float32
print(f"Conv1 weights file size: {file_size} bytes")
print(f"Expected size: {expected_size} bytes (32*3*3*3*4)")
print(f"Match: {file_size == expected_size}")

if file_size != expected_size:
    print(f"\n⚠️  SIZE MISMATCH! Difference: {file_size - expected_size} bytes")
    print("This could indicate:")
    print("  - Extra header bytes")
    print("  - Missing data")
    print("  - Wrong data format")
else:
    print("\n✅ File size is correct")

# Read the data
data = np.fromfile(conv1_w_file, dtype=np.float32)
print(f"\nNumber of float32 values: {data.size}")
print(f"Expected: {32*3*3*3}")

# Check for any NaN or Inf values
print(f"\nData quality:")
print(f"  NaN values: {np.isnan(data).sum()}")
print(f"  Inf values: {np.isinf(data).sum()}")
print(f"  Min: {data.min():.6f}")
print(f"  Max: {data.max():.6f}")
print(f"  Mean: {data.mean():.6f}")

# Print first 20 values to see the pattern
print(f"\nFirst 20 values in file:")
print(data[:20])

# Check if data looks reasonable for conv weights
print(f"\nData statistics:")
print(f"  Std: {data.std():.6f}")
print(f"  Values in [-1, 1]: {np.sum((data >= -1) & (data <= 1))} / {data.size}")
