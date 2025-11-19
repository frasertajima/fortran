"""
Test if FC weights need to be transposed.
PyTorch Linear layer might expect weights in a different orientation than Fortran.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
from model_loader import load_v28_model

print("="*70)
print("Testing FC Weight Transpose")
print("="*70)
print()

# Load model normally
model = load_v28_model('datasets/cifar10/saved_models/cifar10', device='cpu')

# Get current FC1 weights
fc1_weights_original = model.fc1.weight.data.clone()
print(f"FC1 weights shape: {fc1_weights_original.shape}")  # Should be (512, 2048)
print(f"First 5 values: {fc1_weights_original.flatten()[:5]}")
print()

# Try transposing FC weights
print("Testing with transposed FC weights...")
model.fc1.weight.data = fc1_weights_original.T
model.fc2.weight.data = model.fc2.weight.data.T
model.fc3.weight.data = model.fc3.weight.data.T

# Test
test_data = np.fromfile('datasets/cifar10/cifar10_data/images_test.bin', dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
test_labels = np.fromfile('datasets/cifar10/cifar10_data/labels_test.bin', dtype=np.int32)

model.eval()
correct = 0
n_test = 1000

with torch.no_grad():
    for i in range(n_test):
        img = test_data[i].reshape((3, 32, 32))
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
        if pred == test_labels[i]:
            correct += 1

accuracy = 100.0 * correct / n_test
print(f"\nAccuracy with TRANSPOSED FC weights: {accuracy:.2f}%")

if accuracy > 50:
    print("✅ FC weights needed transpose!")
else:
    print("❌ Transpose didn't help")
