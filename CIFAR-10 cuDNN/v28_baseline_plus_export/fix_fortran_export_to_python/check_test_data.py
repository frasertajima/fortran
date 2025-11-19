"""
Check if test data is being loaded correctly.
Maybe the issue is with data loading, not weight loading!
"""

import numpy as np

# Load test data the way Python does
test_data = np.fromfile('datasets/cifar10/cifar10_data/images_test.bin', dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T

print("Python test data loading:")
print(f"  Shape: {test_data.shape}")
print(f"  First image shape when reshaped: {test_data[0].reshape((3, 32, 32)).shape}")
print(f"  First 10 values of first image: {test_data[0][:10]}")
print(f"  Value range: [{test_data.min():.6f}, {test_data.max():.6f}]")
print()

# The data should be normalized to [0, 1] or [-1, 1]
# Check if it's in the right range
if test_data.min() >= 0 and test_data.max() <= 1:
    print("✅ Data appears to be normalized to [0, 1]")
elif test_data.min() >= -1 and test_data.max() <= 1:
    print("✅ Data appears to be normalized to [-1, 1]")
else:
    print(f"⚠️  Data range is unusual: [{test_data.min()}, {test_data.max()}]")

# Check first image statistics
first_img = test_data[0].reshape((3, 32, 32))
print()
print("First image statistics:")
print(f"  Mean: {first_img.mean():.6f}")
print(f"  Std: {first_img.std():.6f}")
print(f"  Min: {first_img.min():.6f}")
print(f"  Max: {first_img.max():.6f}")
