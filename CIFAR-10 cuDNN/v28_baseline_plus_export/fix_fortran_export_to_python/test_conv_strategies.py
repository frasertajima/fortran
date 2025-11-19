"""
Test different weight loading strategies and compare Conv1 output with Fortran.

Fortran output (from your test):
  conv1_out[0,0,0,0:4] = 0.194, 0.384, 0.387, 0.390, 0.411
  conv1_out[0,15,15,0] = -0.047

PyTorch output (current):
  conv1_out[0,0,0,0:4] = -0.363, -0.482, -0.516, -0.496, -0.481
  conv1_out[0,0,15,15] = -1.418

These are VERY different, indicating wrong weight loading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Load test image (first test image from CIFAR-10)
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((10000, 3072))
first_image = test_data[0, :]  # Shape: (3072,)

# Reshape to (3, 32, 32) for PyTorch
first_image_chw = first_image.reshape((3, 32, 32), order='C')  # CHW format
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)  # Add batch dim: (1, 3, 32, 32)

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
print(f"Input mean: {input_tensor.mean():.6f}")

# Load conv1 weights and bias
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_data = np.fromfile(model_dir / "conv1_weights.bin", dtype=np.float32)
conv1_b_data = np.fromfile(model_dir / "conv1_bias.bin", dtype=np.float32)

print("\n" + "="*70)
print("STRATEGY 1: Current method - reshape F-order, transpose (0,3,1,2)")
print("="*70)
conv1_w_s1 = conv1_w_data.reshape((32, 3, 3, 3), order='F')
conv1_w_s1 = np.ascontiguousarray(conv1_w_s1)
conv1_w_s1 = np.transpose(conv1_w_s1, (0, 3, 1, 2))

conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
conv.weight.data = torch.from_numpy(conv1_w_s1)
conv.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    output_s1 = conv(input_tensor)

print(f"Output shape: {output_s1.shape}")
print(f"Output[0,0,0,0:5]: {output_s1[0, 0, 0, :5].numpy()}")
print(f"Output[0,0,15,15]: {output_s1[0, 0, 15, 15].item():.6f}")

print("\n" + "="*70)
print("STRATEGY 2: No transpose - just reshape F-order")
print("="*70)
conv1_w_s2 = conv1_w_data.reshape((32, 3, 3, 3), order='F')
conv1_w_s2 = np.ascontiguousarray(conv1_w_s2)

conv2 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
conv2.weight.data = torch.from_numpy(conv1_w_s2)
conv2.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    output_s2 = conv2(input_tensor)

print(f"Output shape: {output_s2.shape}")
print(f"Output[0,0,0,0:5]: {output_s2[0, 0, 0, :5].numpy()}")
print(f"Output[0,0,15,15]: {output_s2[0, 0, 15, 15].item():.6f}")

print("\n" + "="*70)
print("STRATEGY 3: Reshape as (3,3,3,32) F-order, transpose to (32,3,3,3)")
print("="*70)
conv1_w_s3 = conv1_w_data.reshape((3, 3, 3, 32), order='F')
conv1_w_s3 = np.transpose(conv1_w_s3, (3, 2, 0, 1))  # (3,3,3,32) -> (32,3,3,3)
conv1_w_s3 = np.ascontiguousarray(conv1_w_s3)

conv3 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
conv3.weight.data = torch.from_numpy(conv1_w_s3)
conv3.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    output_s3 = conv3(input_tensor)

print(f"Output shape: {output_s3.shape}")
print(f"Output[0,0,0,0:5]: {output_s3[0, 0, 0, :5].numpy()}")
print(f"Output[0,0,15,15]: {output_s3[0, 0, 15, 15].item():.6f}")

print("\n" + "="*70)
print("FORTRAN EXPECTED OUTPUT:")
print("="*70)
print("conv1_out[0,0,0,0:4] = 0.194, 0.384, 0.387, 0.390, 0.411")
print("conv1_out[0,15,15,0] = -0.047")
print("\nNote: Fortran indexing is (W,H,C,N), so [0,0,0,0:4] means:")
print("  W=0 (x=0), H=0 (y=0), C=0 (channel 0), N=0 (batch 0), values at x=0:4")
print("In PyTorch (N,C,H,W): [0,0,0,0:5] means batch=0, channel=0, y=0, x=0:5")
