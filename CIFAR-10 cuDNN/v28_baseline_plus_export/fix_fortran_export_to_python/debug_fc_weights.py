"""
Debug FC weight shapes to understand the issue.
"""

import numpy as np
from pathlib import Path

model_dir = Path("datasets/cifar10/saved_models/cifar10")

# Check FC1 weights
fc1_data = np.fromfile(model_dir / "fc1_weights.bin", dtype=np.float32)
print(f"FC1 raw data size: {fc1_data.size}")
print(f"Expected: 512 * 2048 = {512 * 2048}")
print()

# Try the current loading method
fc1_w = fc1_data.reshape((2048, 512), order='C')
fc1_w = fc1_w.T  # (512, 2048)
print(f"Current method shape: {fc1_w.shape}")
print(f"PyTorch nn.Linear(2048, 512) expects weight shape: (512, 2048)")
print(f"Match: {fc1_w.shape == (512, 2048)}")
print()

# Check what Fortran actually wrote
print("Fortran exports FC weights as (out_features, in_features) in F-order")
print("For FC1: (512, 2048) in F-order")
print()

# Correct loading
fc1_correct = fc1_data.reshape((512, 2048), order='F')
fc1_correct = np.ascontiguousarray(fc1_correct)
print(f"Correct method (reshape F-order): {fc1_correct.shape}")
print()

# Check if they're different
print(f"Are they the same? {np.allclose(fc1_w, fc1_correct)}")
print(f"Max difference: {np.max(np.abs(fc1_w - fc1_correct))}")
