"""
Test the current Fortran binary format (WITH transpose).

This script will:
1. Load the existing weights (which have transpose applied)
2. Test different loading strategies
3. Determine which one gives the best accuracy
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, 'inference')
from model_loader import load_v28_model

# Load test data
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_labels_file = Path("datasets/cifar10/cifar10_data/labels_test.bin")

test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
test_labels = np.fromfile(test_labels_file, dtype=np.int32)

# Use first 1000 images for quick test
test_data_subset = test_data[:1000]
test_labels_subset = test_labels[:1000]

print("="*70)
print("Testing Fortran Binary Format (WITH transpose)")
print("="*70)
print()

# Test current model loader
print("Loading model with current model_loader.py...")
model = load_v28_model('datasets/cifar10/saved_models/cifar10', device='cpu')
model.eval()

correct = 0
with torch.no_grad():
    for i in range(len(test_data_subset)):
        img = test_data_subset[i].reshape((3, 32, 32))
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
        
        if pred == test_labels_subset[i]:
            correct += 1

accuracy = 100.0 * correct / len(test_data_subset)
print(f"Accuracy with current loader: {accuracy:.2f}%")
print(f"Expected: ~78-79%")
print()

if accuracy > 70:
    print("✅ EXCELLENT! Weights are loading correctly!")
elif accuracy > 50:
    print("⚠️  GOOD but not perfect - may need minor adjustments")
elif accuracy > 30:
    print("⚠️  PARTIAL - some weights loading correctly but not all")
else:
    print("❌ POOR - weights not loading correctly")

print()
print("="*70)
print("Binary Format Summary")
print("="*70)
print()
print("Fortran process:")
print("1. Allocate: conv1_weights(32, 3, 3, 3) in F-order")
print("2. Transpose: temp_c_order(3, 3, 3, 32) with element reversal")
print("3. Reshape: reshape(temp_c_order, [32, 3, 3, 3]) in F-order")
print("4. Export: Write to binary as-is")
print()
print("Python loading (current model_loader.py):")
with open('inference/model_loader.py', 'r') as f:
    lines = f.readlines()
    in_4d_section = False
    for i, line in enumerate(lines):
        if '# For 4D arrays' in line:
            in_4d_section = True
        if in_4d_section:
            print(f"  Line {i+1}: {line.rstrip()}")
            if 'elif len(shape) == 1' in line or 'else:' in line:
                break
