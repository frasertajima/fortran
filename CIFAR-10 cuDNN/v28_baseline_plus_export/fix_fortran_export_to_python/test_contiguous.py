"""
Test if ascontiguousarray is causing the problem.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load test data
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_labels_file = Path("datasets/cifar10/cifar10_data/labels_test.bin")

test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
test_labels = np.fromfile(test_labels_file, dtype=np.int32)

model_dir = Path("datasets/cifar10/saved_models/cifar10")

print("Testing F-order WITHOUT ascontiguousarray...")

# Load conv1 weights
conv1_data = np.fromfile(model_dir / 'conv1_weights.bin', dtype=np.float32)
conv1_w = conv1_data.reshape((32, 3, 3, 3), order='F')
# DON'T call ascontiguousarray - keep it F-order!
conv1_w_torch = torch.from_numpy(conv1_w)

print(f"Conv1 weight shape: {conv1_w_torch.shape}")
print(f"Conv1 weight is contiguous: {conv1_w_torch.is_contiguous()}")
print(f"Conv1 weight stride: {conv1_w_torch.stride()}")

# Try to use it in a conv layer
try:
    conv = nn.Conv2d(3, 32, 3, 1, 1)
    conv.weight.data = conv1_w_torch
    
    # Test on one image
    img = test_data[0].reshape((3, 32, 32))
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    
    with torch.no_grad():
        output = conv(img_tensor)
    
    print(f"✅ Conv works with F-order tensor!")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("Now testing WITH ascontiguousarray...")

conv1_w_c = np.ascontiguousarray(conv1_w)
conv1_w_c_torch = torch.from_numpy(conv1_w_c)

print(f"Conv1 weight (C-order) is contiguous: {conv1_w_c_torch.is_contiguous()}")
print(f"Conv1 weight (C-order) stride: {conv1_w_c_torch.stride()}")

try:
    conv2 = nn.Conv2d(3, 32, 3, 1, 1)
    conv2.weight.data = conv1_w_c_torch
    
    with torch.no_grad():
        output2 = conv2(img_tensor)
    
    print(f"✅ Conv works with C-order tensor!")
    print(f"Output mean: {output2.mean().item():.6f}")
    
    # Compare outputs
    if torch.allclose(output, output2):
        print("✅ Outputs are IDENTICAL!")
    else:
        print(f"❌ Outputs are DIFFERENT!")
        print(f"   Max difference: {(output - output2).abs().max().item():.6f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
