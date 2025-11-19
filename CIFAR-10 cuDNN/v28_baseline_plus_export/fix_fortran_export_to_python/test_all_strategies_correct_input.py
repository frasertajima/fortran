"""
Test ALL weight loading strategies with the CORRECT input data.

Now that we have the correct input (mean=0.425), let's test all strategies.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load test data CORRECTLY (accounting for transpose)
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T  # (10000, 3072)

first_image_flat = test_data[0, :]
first_image_chw = first_image_flat.reshape((3, 32, 32), order='C')
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)

print(f"Input mean: {first_image_flat.mean():.6f} (should be ~0.425)")

# Load weights and bias
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_data = np.fromfile(model_dir / "conv1_weights.bin", dtype=np.float32)
conv1_b_data = np.fromfile(model_dir / "conv1_bias.bin", dtype=np.float32)

print("\n" + "="*70)
print("STRATEGY 1: Reshape (32,3,3,3) F-order, NO transpose")
print("="*70)
w1 = conv1_w_data.reshape((32, 3, 3, 3), order='F')
w1 = np.ascontiguousarray(w1)

conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
conv1.weight.data = torch.from_numpy(w1)
conv1.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    out1 = conv1(input_tensor)
print(f"Output[0,0,0,0:5]: {out1[0,0,0,:5].numpy()}")
print(f"Output[0,0,15,15]: {out1[0,0,15,15].item():.6f}")

print("\n" + "="*70)
print("STRATEGY 2: Reshape (3,3,3,32) F-order, transpose (3,2,0,1)")
print("="*70)
w2 = conv1_w_data.reshape((3, 3, 3, 32), order='F')
w2 = np.transpose(w2, (3, 2, 0, 1))  # (3,3,3,32) -> (32,3,3,3)
w2 = np.ascontiguousarray(w2)

conv2 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
conv2.weight.data = torch.from_numpy(w2)
conv2.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    out2 = conv2(input_tensor)
print(f"Output[0,0,0,0:5]: {out2[0,0,0,:5].numpy()}")
print(f"Output[0,0,15,15]: {out2[0,0,15,15].item():.6f}")

print("\n" + "="*70)
print("STRATEGY 3: Reshape (3,3,32,3) F-order, transpose (3,2,1,0)")
print("="*70)
w3 = conv1_w_data.reshape((3, 3, 32, 3), order='F')
w3 = np.transpose(w3, (2, 3, 0, 1))  # (3,3,32,3) -> (32,3,3,3)
w3 = np.ascontiguousarray(w3)

conv3 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
conv3.weight.data = torch.from_numpy(w3)
conv3.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    out3 = conv3(input_tensor)
print(f"Output[0,0,0,0:5]: {out3[0,0,0,:5].numpy()}")
print(f"Output[0,0,15,15]: {out3[0,0,15,15].item():.6f}")

print("\n" + "="*70)
print("STRATEGY 4: Reshape (32,3,3,3) F-order, transpose (0,3,1,2)")
print("="*70)
w4 = conv1_w_data.reshape((32, 3, 3, 3), order='F')
w4 = np.transpose(w4, (0, 3, 1, 2))
w4 = np.ascontiguousarray(w4)

conv4 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
conv4.weight.data = torch.from_numpy(w4)
conv4.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    out4 = conv4(input_tensor)
print(f"Output[0,0,0,0:5]: {out4[0,0,0,:5].numpy()}")
print(f"Output[0,0,15,15]: {out4[0,0,15,15].item():.6f}")

print("\n" + "="*70)
print("FORTRAN TARGET:")
print("="*70)
print("Output[0,0,0,0:5]: [0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115]")
print("Output[0,15,15,0]: -0.04665595")
