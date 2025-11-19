"""
Compare how Fortran and Python load and process the first test image.
This will help identify if there's a data loading mismatch.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
from model_loader import load_v28_model

print("="*70)
print("🔍 Comparing Fortran vs Python Data Loading")
print("="*70)
print()

# Load test data the Python way
print("Loading test data (Python method)...")
test_data = np.fromfile('datasets/cifar10/cifar10_data/images_test.bin', dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
test_labels = np.fromfile('datasets/cifar10/cifar10_data/labels_test.bin', dtype=np.int32)

print(f"  Shape: {test_data.shape}")
print(f"  First image label: {test_labels[0]}")
print(f"  First 10 values: {test_data[0, :10]}")
print()

# Get first image
first_img = test_data[0].reshape((3, 32, 32))
print("First image (Python):")
print(f"  Shape: {first_img.shape}")
print(f"  Min: {first_img.min():.6f}, Max: {first_img.max():.6f}, Mean: {first_img.mean():.6f}")
print(f"  Channel 0 mean: {first_img[0].mean():.6f}")
print(f"  Channel 1 mean: {first_img[1].mean():.6f}")
print(f"  Channel 2 mean: {first_img[2].mean():.6f}")
print()

# Load model and run inference
print("Loading model...")
model = load_v28_model('datasets/cifar10/saved_models/cifar10', device='cpu')
model.eval()

print("Running inference...")
with torch.no_grad():
    img_tensor = torch.from_numpy(first_img).unsqueeze(0)
    output = model(img_tensor)
    pred = output.argmax(dim=1).item()
    logits = output[0].numpy()

print(f"  Prediction: {pred}")
print(f"  Confidence: {torch.softmax(output, dim=1).max().item():.4f}")
print(f"  Logits: {logits}")
print()

print("="*70)
print("Summary:")
print(f"  Python loads first image with label: {test_labels[0]}")
print(f"  Python model predicts: {pred}")
print(f"  Match: {'✅ YES' if pred == test_labels[0] else '❌ NO'}")
print("="*70)
