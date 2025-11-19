"""
FINAL TEST: Account for the data transpose in prepare_cifar10.py

The data is written as (3072, N).T which means it's transposed!
So we need to read it correctly.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
from pathlib import Path
from model_loader import load_v28_model

print("="*70)
print("FINAL TEST with correct data loading")
print("="*70)

# Load the model
model = load_v28_model('datasets/cifar10/saved_models/cifar10',
                       in_channels=3, num_classes=10, input_size=32, device='cpu')

# Load test data - accounting for the transpose!
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)

# The file was written as (3072, 10000).tofile()
# So we read it back as (3072, 10000) and transpose to get (10000, 3072)
test_data = test_data.reshape((3072, 10000), order='C')
test_data = test_data.T  # Now (10000, 3072)

print(f"Test data shape: {test_data.shape}")

# Get first test image
first_image_flat = test_data[0, :]  # (3072,)
print(f"First image mean: {first_image_flat.mean():.6f}")
print(f"First image range: [{first_image_flat.min():.4f}, {first_image_flat.max():.4f}]")

# Reshape to (C, H, W) = (3, 32, 32)
first_image_chw = first_image_flat.reshape((3, 32, 32), order='C')
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)  # (1, 3, 32, 32)

print(f"\nInput tensor shape: {input_tensor.shape}")

# Run just Conv1 + bias
with torch.no_grad():
    conv1_out = model.conv1(input_tensor)

print("\n" + "="*70)
print("PyTorch Conv1 Output:")
print("="*70)
print(f"Shape: {conv1_out.shape}")
print(f"Output[0,0,0,0:5]: {conv1_out[0, 0, 0, :5].numpy()}")
print(f"Output[0,0,15,15]: {conv1_out[0, 0, 15, 15].item():.6f}")
print(f"Mean: {conv1_out.mean().item():.6f}")
print(f"Range: [{conv1_out.min().item():.6f}, {conv1_out.max().item():.6f}]")

print("\n" + "="*70)
print("Fortran Conv1 Output (expected):")
print("="*70)
print("conv1_out[0,0,0,0:4] = 0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115")
print("conv1_out[0,15,15,0] = -0.04665595")
print("Mean: -0.0871991")
print("Range: [-2.330830, 1.196022]")

print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
fortran_vals = np.array([0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115])
pytorch_vals = conv1_out[0, 0, 0, :5].numpy()

print(f"Fortran [0,0,0,0:5]: {fortran_vals}")
print(f"PyTorch [0,0,0,0:5]: {pytorch_vals}")
print(f"Difference: {np.abs(fortran_vals - pytorch_vals)}")
print(f"Max difference: {np.max(np.abs(fortran_vals - pytorch_vals)):.6f}")

if np.max(np.abs(fortran_vals - pytorch_vals)) < 0.01:
    print("\n✅ SUCCESS! Outputs MATCH!")
else:
    print("\n❌ Still not matching.")
