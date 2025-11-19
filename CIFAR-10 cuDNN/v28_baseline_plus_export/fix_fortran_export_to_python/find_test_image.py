"""
Debug: Check if we're using the same input image as Fortran.

Fortran reports: Mean: 0.4250444
Python reports: Mean: 0.513262

These are DIFFERENT! We need to find the correct input.
"""

import numpy as np
from pathlib import Path

# Load test data
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((10000, 3072))

print("Checking first few test images:")
for i in range(5):
    img = test_data[i, :]
    print(f"Image {i}: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.6f}")

print("\nFortran reported input mean: 0.4250444")
print("Looking for image with matching mean...")

# Find image with mean closest to 0.4250444
target_mean = 0.4250444
best_idx = -1
best_diff = float('inf')

for i in range(min(100, test_data.shape[0])):
    img_mean = test_data[i, :].mean()
    diff = abs(img_mean - target_mean)
    if diff < best_diff:
        best_diff = diff
        best_idx = i

print(f"\nClosest match: Image {best_idx} with mean={test_data[best_idx, :].mean():.6f}")
print(f"Difference from Fortran: {best_diff:.6f}")

if best_diff < 0.001:
    print(f"✅ Found matching image at index {best_idx}!")
else:
    print(f"❌ No exact match found. Fortran might be using different data.")
