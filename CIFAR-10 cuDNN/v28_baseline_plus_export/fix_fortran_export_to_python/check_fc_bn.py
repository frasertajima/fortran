"""
Check if FC weights are loading correctly.
Conv weights are confirmed working, so issue must be in FC or BatchNorm.
"""

import numpy as np

print("="*70)
print("Checking FC Weight Loading")
print("="*70)
print()

# Load FC1 weights
fc1_data = np.fromfile('datasets/cifar10/saved_models/cifar10/fc1_weights.bin', dtype=np.float32)
print(f"FC1 binary file size: {len(fc1_data)} elements")
print(f"Expected: 2048 * 512 = {2048 * 512}")
print()

# Try F-order reshape
fc1_f = fc1_data.reshape((512, 2048), order='F')
print(f"F-order reshape: {fc1_f.shape}")
print(f"  First 5 values: {fc1_f.flatten()[:5]}")
print(f"  Value range: [{fc1_f.min():.6f}, {fc1_f.max():.6f}]")
print()

# Try C-order reshape  
fc1_c = fc1_data.reshape((512, 2048), order='C')
print(f"C-order reshape: {fc1_c.shape}")
print(f"  First 5 values: {fc1_c.flatten()[:5]}")
print(f"  Value range: [{fc1_c.min():.6f}, {fc1_c.max():.6f}]")
print()

# Check if they're different
if np.allclose(fc1_f, fc1_c):
    print("⚠️  F-order and C-order give SAME result - something is wrong!")
else:
    print("✅ F-order and C-order give different results (as expected)")

print()
print("="*70)
print("Checking BatchNorm Parameters")
print("="*70)
print()

# Check BN1
bn1_scale = np.fromfile('datasets/cifar10/saved_models/cifar10/bn1_scale.bin', dtype=np.float32)
bn1_bias = np.fromfile('datasets/cifar10/saved_models/cifar10/bn1_bias.bin', dtype=np.float32)
bn1_mean = np.fromfile('datasets/cifar10/saved_models/cifar10/bn1_running_mean.bin', dtype=np.float32)
bn1_var = np.fromfile('datasets/cifar10/saved_models/cifar10/bn1_running_var.bin', dtype=np.float32)

print(f"BN1 scale: {len(bn1_scale)} elements, range [{bn1_scale.min():.6f}, {bn1_scale.max():.6f}]")
print(f"BN1 bias: {len(bn1_bias)} elements, range [{bn1_bias.min():.6f}, {bn1_bias.max():.6f}]")
print(f"BN1 mean: {len(bn1_mean)} elements, range [{bn1_mean.min():.6f}, {bn1_mean.max():.6f}]")
print(f"BN1 var: {len(bn1_var)} elements, range [{bn1_var.min():.6f}, {bn1_var.max():.6f}]")

# Check if variance is reasonable (should be > 0)
if np.any(bn1_var <= 0):
    print("❌ ERROR: BN1 variance has non-positive values!")
else:
    print("✅ BN1 variance all positive")

print()
print("First 5 BN1 values:")
print(f"  Scale: {bn1_scale[:5]}")
print(f"  Bias: {bn1_bias[:5]}")
print(f"  Mean: {bn1_mean[:5]}")
print(f"  Var: {bn1_var[:5]}")
