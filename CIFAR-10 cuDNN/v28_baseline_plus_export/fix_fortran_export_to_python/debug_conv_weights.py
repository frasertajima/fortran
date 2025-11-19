"""
Debug: Check if C-order export is actually working for conv weights
"""

import numpy as np

# Load conv1 weights
conv1_data = np.fromfile('datasets/cifar10/saved_models/cifar10/conv1_weights.bin', dtype=np.float32)

print("Conv1 weights file:")
print(f"  Total elements: {len(conv1_data)}")
print(f"  Expected: 32 * 3 * 3 * 3 = {32*3*3*3}")
print(f"  First 20 values:")
for i in range(20):
    print(f"    [{i}]: {conv1_data[i]:.6f}")

print()
print("Testing different reshape strategies:")
print()

# Strategy 1: C-order reshape (what we're currently using)
print("Strategy 1: C-order reshape (32,3,3,3)")
arr_c = conv1_data.reshape((32,3,3,3), order='C')
print(f"  arr[0,0,0,0] = {arr_c[0,0,0,0]:.6f}")
print(f"  arr[0,0,0,1] = {arr_c[0,0,0,1]:.6f}")
print(f"  arr[0,0,1,0] = {arr_c[0,0,1,0]:.6f}")
print()

# Strategy 2: F-order reshape
print("Strategy 2: F-order reshape (32,3,3,3)")
arr_f = conv1_data.reshape((32,3,3,3), order='F')
print(f"  arr[0,0,0,0] = {arr_f[0,0,0,0]:.6f}")
print(f"  arr[0,0,0,1] = {arr_f[0,0,0,1]:.6f}")
print(f"  arr[0,0,1,0] = {arr_f[0,0,1,0]:.6f}")
print()

# Check if values look reasonable (should be small, around -0.2 to 0.2 for He init)
print("Value range check:")
print(f"  C-order: min={arr_c.min():.6f}, max={arr_c.max():.6f}, mean={arr_c.mean():.6f}")
print(f"  F-order: min={arr_f.min():.6f}, max={arr_f.max():.6f}, mean={arr_f.mean():.6f}")
