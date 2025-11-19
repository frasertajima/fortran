"""
Verify Channel 24 match - check if it's consistently close.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load test data
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
first_image_flat = test_data[0, :]
first_image_chw = first_image_flat.reshape((3, 32, 32), order='C')
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)

# Load weights - Strategy 2
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_data = np.fromfile(model_dir / "conv1_weights.bin", dtype=np.float32)
conv1_b_data = np.fromfile(model_dir / "conv1_bias.bin", dtype=np.float32)

w = conv1_w_data.reshape((3, 3, 3, 32), order='F')
w = np.transpose(w, (3, 2, 0, 1))
w = np.ascontiguousarray(w)

conv = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
conv.weight.data = torch.from_numpy(w)
conv.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    out = conv(input_tensor)

print("Best match: Channel 24, H=0, W=0")
print()

# Check the first 5 values
print("First 5 values:")
print(f"  PyTorch [0,24,0,0:5]: {out[0,24,0,0:5].numpy()}")
print(f"  Fortran target:       [0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115]")
print(f"  Difference:           {np.abs(out[0,24,0,0:5].numpy() - np.array([0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115]))}")
print()

# Check the [15,15] value
print("Value at [15,15]:")
print(f"  PyTorch [0,24,15,15]: {out[0,24,15,15].item():.6f}")
print(f"  Fortran target:       -0.046656")
print(f"  Difference:           {abs(out[0,24,15,15].item() - (-0.046656)):.6f}")
print()

# Check overall statistics
print("Overall statistics:")
print(f"  PyTorch mean: {out[0,24,:,:].mean().item():.6f}")
print(f"  Fortran mean: -0.087199")
print(f"  PyTorch range: [{out[0,24,:,:].min().item():.6f}, {out[0,24,:,:].max().item():.6f}]")
print(f"  Fortran range: [-2.330830, 1.196022]")
print()

print("Conclusion:")
if abs(out[0,24,0,0].item() - 0.1936478) < 0.05:
    print("✅ Channel 24 is a VERY CLOSE match!")
    print("The small difference might be due to:")
    print("  1. Floating point precision differences")
    print("  2. cuDNN vs PyTorch conv implementation differences")
    print("  3. Minor numerical errors in weight loading")
    print()
    print("This suggests Strategy 2 is CORRECT:")
    print("  - Reshape as (3,3,3,32) with order='F'")
    print("  - Transpose (3,2,0,1) to get (32,3,3,3)")
    print("  - But output channel mapping might be permuted!")
else:
    print("❌ Still not a good match")
