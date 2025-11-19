"""
Quick accuracy test on 1000 test images.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
from pathlib import Path
from model_loader import load_v28_model

# Load model
model = load_v28_model('datasets/cifar10/saved_models/cifar10', device='cpu')
model.eval()

# Load test data
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_labels_file = Path("datasets/cifar10/cifar10_data/labels_test.bin")

test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T

test_labels = np.fromfile(test_labels_file, dtype=np.int32)

# Test on first 1000 images
n_test = 1000
correct = 0

print(f"Testing on {n_test} images...")

with torch.no_grad():
    for i in range(n_test):
        img = test_data[i].reshape((3, 32, 32))
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
        
        if pred == test_labels[i]:
            correct += 1
        
        if (i + 1) % 100 == 0:
            acc = 100.0 * correct / (i + 1)
            print(f"  {i+1}/{n_test}: {acc:.2f}%")

accuracy = 100.0 * correct / n_test
print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{n_test})")
print(f"Expected: ~78-79% (matching Fortran training accuracy)")

if accuracy > 70:
    print("✅ EXCELLENT! Accuracy matches Fortran training!")
elif accuracy > 50:
    print("⚠️  Good but not great - some improvement needed")
else:
    print("❌ Still too low - weights not loading correctly")
