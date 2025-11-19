"""
Compare Fortran vs Python input loading - fixed version.
"""

import numpy as np

print("="*70)
print("🔍 Comparing Fortran vs Python Input Loading")
print("="*70)
print()

# Load Fortran's exported input
print("Loading Fortran input...")
fortran_input = np.fromfile('datasets/cifar10/saved_models/cifar10/debug_input.bin', dtype=np.float32)
print(f"  Raw size: {len(fortran_input)}")

# Reshape - Fortran exports as (W, H, C, N) in F-order
fortran_img = fortran_input.reshape((32, 32, 3), order='F')
print(f"  Reshaped to (W,H,C): {fortran_img.shape}")
print(f"  Min: {fortran_img.min():.6f}, Max: {fortran_img.max():.6f}, Mean: {fortran_img.mean():.6f}")
print(f"  First 10 flat values: {fortran_input[:10]}")
print()

# Load Python input
print("Loading Python input...")
test_data = np.fromfile('datasets/cifar10/cifar10_data/images_test.bin', dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
python_flat = test_data[0]
python_img = python_flat.reshape((3, 32, 32))
print(f"  Shape (C,H,W): {python_img.shape}")
print(f"  Min: {python_img.min():.6f}, Max: {python_img.max():.6f}, Mean: {python_img.mean():.6f}")
print(f"  First 10 flat values: {python_flat[:10]}")
print()

# Compare raw flat arrays
print("="*70)
print("Raw Flat Array Comparison")
print("="*70)
match_flat = np.allclose(fortran_input, python_flat, atol=1e-6)
print(f"Direct flat comparison: {'✅ MATCH' if match_flat else '❌ No match'}")
print()

if not match_flat:
    print("Inputs are DIFFERENT!")
    print(f"  Fortran mean: {fortran_input.mean():.6f}")
    print(f"  Python mean:  {python_flat.mean():.6f}")
    print(f"  Difference: {abs(fortran_input.mean() - python_flat.mean()):.6f}")
    print()
    print("This explains the 10% accuracy - Fortran and Python are loading different data!")
else:
    print("Inputs are the same - bug must be elsewhere")

print("="*70)
