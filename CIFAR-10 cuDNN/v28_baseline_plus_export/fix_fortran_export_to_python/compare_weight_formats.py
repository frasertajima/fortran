"""
Compare Fortran exported weights with PyTorch expected format.

This will help us understand the correct transpose operation.
"""

import numpy as np
from pathlib import Path

# Load Fortran weights
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_weights_file = model_dir / "conv1_weights.bin"

data = np.fromfile(conv1_weights_file, dtype=np.float32)

print("="*70)
print("CURRENT LOADING METHOD (from model_loader.py)")
print("="*70)
# Current method in model_loader.py
weights_current = data.reshape((32, 3, 3, 3), order='F')
weights_current = np.ascontiguousarray(weights_current)
weights_current_transposed = np.transpose(weights_current, (0, 3, 1, 2))

print(f"After reshape with order='F': shape = {weights_current.shape}")
print(f"After transpose (0,3,1,2): shape = {weights_current_transposed.shape}")
print(f"\nFirst filter [0], first channel [0], 3x3:")
print(weights_current_transposed[0, 0, :, :])

print("\n" + "="*70)
print("ALTERNATIVE: What if cuDNN stores in (K,C,H,W) already?")
print("="*70)
# Maybe cuDNN already has it in the right format?
weights_alt = data.reshape((32, 3, 3, 3), order='F')
print(f"Shape: {weights_alt.shape}")
print(f"First filter [0], first channel [0], 3x3:")
print(weights_alt[0, 0, :, :])

print("\n" + "="*70)
print("ALTERNATIVE 2: Maybe we need different transpose?")
print("="*70)
# Try (3,2,1,0) transpose
weights_alt2 = data.reshape((32, 3, 3, 3), order='F')
weights_alt2_transposed = np.transpose(weights_alt2, (3, 2, 1, 0))
print(f"After transpose (3,2,1,0): shape = {weights_alt2_transposed.shape}")
print(f"First filter [0], first channel [0], 3x3:")
print(weights_alt2_transposed[0, 0, :, :])

print("\n" + "="*70)
print("PyTorch Conv2d expected format:")
print("="*70)
print("PyTorch expects: (out_channels, in_channels, kernel_h, kernel_w)")
print("For Conv1: (32, 3, 3, 3)")
print("\nThe question is: which interpretation matches the Fortran cuDNN output?")
