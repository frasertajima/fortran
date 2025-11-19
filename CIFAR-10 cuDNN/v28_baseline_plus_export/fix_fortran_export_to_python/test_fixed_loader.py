"""
Test the FIXED model_loader.py against Fortran output.

Expected Fortran output:
  conv1_out[0,0,0,0:4] = 0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115
  conv1_out[0,15,15,0] = -0.04665595
  Mean: -0.0871991
  Range: [-2.330830, 1.196022]
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
from pathlib import Path
from model_loader import load_v28_model

print("="*70)
print("Testing FIXED model_loader.py")
print("="*70)

# Load the model with the FIXED loader
model = load_v28_model('datasets/cifar10/saved_models/cifar10',
                       in_channels=3, num_classes=10, input_size=32, device='cpu')

# Load test image - same as Fortran uses
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((10000, 3072))
first_image_flat = test_data[0, :]

# Reshape to (C, H, W) = (3, 32, 32) - CHW format
first_image_chw = first_image_flat.reshape((3, 32, 32), order='C')
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)  # (1, 3, 32, 32)

print(f"\nInput tensor shape: {input_tensor.shape}")
print(f"Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
print(f"Input mean: {input_tensor.mean():.6f}")

# Run just Conv1 + bias (no BN, no activation)
with torch.no_grad():
    conv1_out = model.conv1(input_tensor)

print("\n" + "="*70)
print("PyTorch Conv1 Output (after conv+bias only):")
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

fortran_val_15_15 = -0.04665595
pytorch_val_15_15 = conv1_out[0, 0, 15, 15].item()
print(f"\nFortran [0,15,15,0]: {fortran_val_15_15:.6f}")
print(f"PyTorch [0,0,15,15]: {pytorch_val_15_15:.6f}")
print(f"Difference: {abs(fortran_val_15_15 - pytorch_val_15_15):.6f}")

# Check if values match within tolerance
if np.max(np.abs(fortran_vals - pytorch_vals)) < 0.01:
    print("\n✅ SUCCESS! Conv1 outputs MATCH!")
    print("The model loader is now correctly loading Fortran weights!")
else:
    print("\n❌ Still not matching. Further investigation needed.")
    print("\nPossible issues:")
    print("1. Different test image being used")
    print("2. Weight loading still has issues")
    print("3. Input format mismatch")
