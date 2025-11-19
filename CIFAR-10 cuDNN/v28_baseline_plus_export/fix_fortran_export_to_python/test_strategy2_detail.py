"""
Deep dive into Strategy 2 - check multiple positions and channels.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load test data CORRECTLY
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
first_image_flat = test_data[0, :]
first_image_chw = first_image_flat.reshape((3, 32, 32), order='C')
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)

# Load weights
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_data = np.fromfile(model_dir / "conv1_weights.bin", dtype=np.float32)
conv1_b_data = np.fromfile(model_dir / "conv1_bias.bin", dtype=np.float32)

# Strategy 2
w2 = conv1_w_data.reshape((3, 3, 3, 32), order='F')
w2 = np.transpose(w2, (3, 2, 0, 1))  # (3,3,3,32) -> (32,3,3,3)
w2 = np.ascontiguousarray(w2)

conv2 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
conv2.weight.data = torch.from_numpy(w2)
conv2.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    out2 = conv2(input_tensor)

print("Strategy 2 detailed output:")
print(f"Channel 0, position (0, 0:10): {out2[0, 0, 0, :10].numpy()}")
print(f"Channel 1, position (0, 0:10): {out2[0, 1, 0, :10].numpy()}")
print(f"Channel 0, position (15, 15): {out2[0, 0, 15, 15].item():.6f}")

print("\nFortran output:")
print("conv1_out[0,0,0,0:4] = 0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115")
print("conv1_out[0,15,15,0] = -0.04665595")

print("\nLet's check if maybe it's a different channel:")
for c in range(min(5, 32)):
    print(f"Channel {c}, position (0, 0:5): {out2[0, c, 0, :5].numpy()}")
