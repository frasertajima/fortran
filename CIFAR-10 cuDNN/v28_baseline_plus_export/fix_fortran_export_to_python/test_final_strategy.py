"""
Final test: Use EXACT same input format as Fortran and test Strategy 3.

Fortran uses: pixel_idx = (c-1)*H*W + (h-1)*W + w
This is CHW order (channel-major).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load test image - SAME as Fortran
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((10000, 3072))
first_image_flat = test_data[0, :]  # Shape: (3072,) - flat CHW format

print(f"Input stats (flat):")
print(f"  Min: {first_image_flat.min():.6f}")
print(f"  Max: {first_image_flat.max():.6f}")
print(f"  Mean: {first_image_flat.mean():.6f}")

# Reshape to (C, H, W) = (3, 32, 32) - this matches Fortran's CHW storage
first_image_chw = first_image_flat.reshape((3, 32, 32), order='C')
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)  # (1, 3, 32, 32)

print(f"\nInput tensor shape: {input_tensor.shape}")
print(f"Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
print(f"Input mean: {input_tensor.mean():.6f}")

# Load conv1 weights and bias
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_data = np.fromfile(model_dir / "conv1_weights.bin", dtype=np.float32)
conv1_b_data = np.fromfile(model_dir / "conv1_bias.bin", dtype=np.float32)

print("\n" + "="*70)
print("STRATEGY 3: Reshape as (3,3,3,32) F-order, transpose to (32,3,3,3)")
print("="*70)
# Fortran writes (32,3,3,3) in F-order
# When read as (3,3,3,32) F-order and transposed, we get correct layout
conv1_w = conv1_w_data.reshape((3, 3, 3, 32), order='F')
conv1_w = np.transpose(conv1_w, (3, 2, 0, 1))  # (3,3,3,32) -> (32,3,3,3)
conv1_w = np.ascontiguousarray(conv1_w)

conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
conv.weight.data = torch.from_numpy(conv1_w)
conv.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    output = conv(input_tensor)

print(f"Output shape: {output.shape}")
print(f"Output[0,0,0,0:5]: {output[0, 0, 0, :5].numpy()}")
print(f"Output[0,0,15,15]: {output[0, 0, 15, 15].item():.6f}")
print(f"Output mean: {output.mean().item():.6f}")
print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

print("\n" + "="*70)
print("FORTRAN EXPECTED OUTPUT:")
print("="*70)
print("conv1_out[0,0,0,0:4] = 0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115")
print("conv1_out[0,15,15,0] = -0.04665595")
print("Mean: -0.0871991")
print("Range: [-2.330830, 1.196022]")

print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
fortran_vals = np.array([0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115])
pytorch_vals = output[0, 0, 0, :5].numpy()
print(f"Fortran [0,0,0,0:5]: {fortran_vals}")
print(f"PyTorch [0,0,0,0:5]: {pytorch_vals}")
print(f"Difference: {np.abs(fortran_vals - pytorch_vals)}")
print(f"Max diff: {np.max(np.abs(fortran_vals - pytorch_vals)):.6f}")

if np.max(np.abs(fortran_vals - pytorch_vals)) < 0.01:
    print("\n✅ MATCH! Strategy 3 is correct!")
else:
    print("\n❌ Still not matching. Need to investigate further.")
