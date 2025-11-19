"""
Test script to verify conv weight loading from Fortran binary.

This will help us understand the actual memory layout.
"""

import numpy as np
from pathlib import Path

# Path to the exported weights
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_weights_file = model_dir / "conv1_weights.bin"

if not conv1_weights_file.exists():
    print(f"❌ File not found: {conv1_weights_file}")
    exit(1)

# Read the raw binary data
data = np.fromfile(conv1_weights_file, dtype=np.float32)
print(f"Total elements: {data.size}")
print(f"Expected: 32 * 3 * 3 * 3 = {32*3*3*3}")

# Try different reshape interpretations
print("\n" + "="*70)
print("INTERPRETATION 1: Fortran shape (32, 3, 3, 3) with F-order")
print("="*70)
weights_f = data.reshape((32, 3, 3, 3), order='F')
print(f"Shape: {weights_f.shape}")
print(f"First filter, first channel, top-left 3x3:")
print(weights_f[0, 0, :, :])
print(f"\nFirst 10 values in memory order:")
print(data[:10])
print(f"First 10 values from reshaped array (flattened in C-order):")
print(weights_f.ravel(order='C')[:10])
print(f"First 10 values from reshaped array (flattened in F-order):")
print(weights_f.ravel(order='F')[:10])

print("\n" + "="*70)
print("INTERPRETATION 2: Fortran shape (3, 3, 3, 32) with F-order")
print("="*70)
weights_f2 = data.reshape((3, 3, 3, 32), order='F')
print(f"Shape: {weights_f2.shape}")
print(f"First 3x3 for first channel, first filter:")
print(weights_f2[:, :, 0, 0])

print("\n" + "="*70)
print("INTERPRETATION 3: Direct C-order reshape to (32, 3, 3, 3)")
print("="*70)
weights_c = data.reshape((32, 3, 3, 3), order='C')
print(f"Shape: {weights_c.shape}")
print(f"First filter, first channel, top-left 3x3:")
print(weights_c[0, 0, :, :])

print("\n" + "="*70)
print("File size check:")
print("="*70)
file_size = conv1_weights_file.stat().st_size
print(f"File size: {file_size} bytes")
print(f"Expected: {32*3*3*3*4} bytes (32*3*3*3 floats * 4 bytes)")
print(f"Match: {file_size == 32*3*3*3*4}")
