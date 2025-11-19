"""
Test Suspect #1: Spatial Flip (Convolution vs Cross-Correlation)

Mathematical convolution rotates kernels 180°, PyTorch uses cross-correlation (no rotation).
If Fortran used true convolution, weights are "upside down".
"""

import sys
sys.path.insert(0, 'inference')

import torch
import numpy as np
from model_loader import load_v28_model

def test_spatial_flip(model_dir, test_data, test_labels, n_test=1000):
    """
    Test if flipping conv kernels 180° fixes the accuracy.
    """
    print("="*70)
    print("🔄 Testing Spatial Flip (Rot180) on Conv Kernels")
    print("="*70)
    print()
    
    # Load model normally
    model = load_v28_model(model_dir, device='cpu')
    
    # Test baseline
    print("Testing BASELINE (no flip)...")
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(n_test):
            img = test_data[i].reshape((3, 32, 32))
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            if pred == test_labels[i]:
                correct += 1
    
    baseline_acc = 100.0 * correct / n_test
    print(f"  Baseline Accuracy: {baseline_acc:.2f}%")
    print()
    
    # Apply spatial flip to ALL conv layers
    print("Applying 180° rotation to all conv kernels...")
    with torch.no_grad():
        # Flip dimensions 2 and 3 (Height and Width)
        model.conv1.weight.data = torch.flip(model.conv1.weight.data, [2, 3])
        model.conv2.weight.data = torch.flip(model.conv2.weight.data, [2, 3])
        model.conv3.weight.data = torch.flip(model.conv3.weight.data, [2, 3])
    print("  ✓ Conv1 weights flipped")
    print("  ✓ Conv2 weights flipped")
    print("  ✓ Conv3 weights flipped")
    print()
    
    # Test with flipped kernels
    print("Testing WITH SPATIAL FLIP...")
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(n_test):
            img = test_data[i].reshape((3, 32, 32))
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            if pred == test_labels[i]:
                correct += 1
    
    flipped_acc = 100.0 * correct / n_test
    print(f"  Flipped Accuracy: {flipped_acc:.2f}%")
    print()
    
    # Results
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline:      {baseline_acc:5.2f}%")
    print(f"Spatial Flip:  {flipped_acc:5.2f}%")
    print(f"Improvement:   {flipped_acc - baseline_acc:+5.2f}%")
    print()
    
    if flipped_acc > 70:
        print("🎉 SUCCESS! Spatial flip fixed it!")
        print("   The issue was Convolution vs Cross-Correlation!")
    elif flipped_acc > baseline_acc + 5:
        print("🔶 Partial improvement - spatial flip helps but isn't the full solution")
    else:
        print("❌ Spatial flip didn't help - this isn't the issue")
    print("="*70)
    
    return baseline_acc, flipped_acc

if __name__ == "__main__":
    # Load test data
    test_data = np.fromfile('datasets/cifar10/cifar10_data/images_test.bin', dtype=np.float32)
    test_data = test_data.reshape((3072, 10000), order='C').T
    test_labels = np.fromfile('datasets/cifar10/cifar10_data/labels_test.bin', dtype=np.int32)
    
    test_spatial_flip("datasets/cifar10/saved_models/cifar10/", test_data, test_labels, n_test=1000)
