#!/usr/bin/env python3
"""
Systematically test all possible dimension permutations to find which matches Fortran output.

Expected Fortran values:
  conv1_out[0,0,0,0:4] = 0.1936, 0.3839, 0.3868, 0.3895, 0.4106
  conv1_out[0,15,15,0] = -0.0467
"""

import sys

sys.path.insert(0, "inference")
from itertools import permutations

import numpy as np
import torch
from model_loader import V28_CNN

# Load test image
test_images = np.fromfile(
    "v28_baseline/datasets/cifar10/cifar10_data/images_test.bin", dtype=np.float32
)
test_images = test_images.reshape((3072, 10000), order="C")
first_image = test_images[:, 0].reshape((3, 32, 32), order="C")
input_tensor = torch.from_numpy(first_image).unsqueeze(0)

# Load weights
conv1_weights = np.fromfile(
    "v28_baseline/datasets/cifar10/saved_models/cifar10/conv1_weights.bin",
    dtype=np.float32,
)
conv1_bias = np.fromfile(
    "v28_baseline/datasets/cifar10/saved_models/cifar10/conv1_bias.bin",
    dtype=np.float32,
)

print("Testing all possible reshape orders and transposes...")
print("=" * 70)
print(f"Weight array size: {conv1_weights.size} (should be {32 * 3 * 3 * 3})")
print()

# Expected from Fortran
expected_vals = np.array([0.1936, 0.3839, 0.3868, 0.3895, 0.4106])
expected_15_15 = -0.0467

best_match = None
best_error = float("inf")
best_config = None

# Try different initial shapes and orders
for init_shape in [(32, 3, 3, 3), (3, 3, 3, 32), (3, 32, 3, 3), (3, 3, 32, 3)]:
    for order in ["F", "C"]:
        for perm in permutations([0, 1, 2, 3]):
            # Load with specific shape and order
            try:
                weights = conv1_weights.reshape(init_shape, order=order)
            except:
                continue

            # Apply transpose
            weights_t = np.transpose(weights, perm)

            # Skip if shape is wrong
            if weights_t.shape != (32, 3, 3, 3):
                continue

            # Create model and test
            model = V28_CNN(
                in_channels=3, num_classes=10, input_size=32, flatten_size=2048
            )
            model.conv1.weight.data = torch.from_numpy(np.ascontiguousarray(weights_t))
            model.conv1.bias.data = torch.from_numpy(conv1_bias)
            model.eval()

            with torch.no_grad():
                conv1_out = model.conv1(input_tensor)

            # Check values
            vals = conv1_out[0, 0, 0, :5].numpy()
            val_15_15 = conv1_out[0, 0, 15, 15].item()

            error = np.abs(vals - expected_vals).sum() + abs(val_15_15 - expected_15_15)

            if error < best_error:
                best_error = error
                best_match = perm
                best_config = (init_shape, order)

                print(
                    f"New best! Shape={init_shape}, order='{order}', transpose={perm}"
                )
                print(f"  Error: {error:.4f}")
                print(f"  Values: {vals}")
                print(f"  [0,15,15]: {val_15_15:.4f}")
                print()

            if error < 0.01:
                print(f"  ✅ PERFECT MATCH FOUND!")
                print(f"  Shape={init_shape}, order='{order}', transpose={perm}")
                break

        if best_error < 0.01:
            break

    if best_error < 0.01:
        break

if best_match:
    print("=" * 70)
    print(f"Best match: transpose{best_match} with error={best_error:.6f}")
else:
    print("=" * 70)
    print("No good match found!")
