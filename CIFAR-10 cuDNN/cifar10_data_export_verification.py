#!/usr/bin/env python3
"""
cifar10_data_export_verification.py
Export CIFAR-10 data to binary files for Fortran verification
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# CIFAR-10 normalization constants (matching PyTorch defaults)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

def export_cifar10_binary():
    print("📥 Loading CIFAR-10 dataset...")
    
    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    
    # Convert to numpy arrays
    print("🔄 Converting training data...")
    x_train = []
    y_train = []
    for img, label in trainset:
        x_train.append(img.numpy())
        y_train.append(label)
    
    x_train = np.array(x_train)  # Shape: (50000, 3, 32, 32)
    y_train = np.array(y_train, dtype=np.float32)  # Shape: (50000,)
    
    print("🔄 Converting test data...")
    x_test = []
    y_test = []
    for img, label in testset:
        x_test.append(img.numpy())
        y_test.append(label)
    
    x_test = np.array(x_test)  # Shape: (10000, 3, 32, 32)
    y_test = np.array(y_test, dtype=np.float32)  # Shape: (10000,)
    
    # Flatten images for Fortran (samples, channels, height, width) -> flat array
    x_train_flat = x_train.flatten().astype(np.float32)
    x_test_flat = x_test.flatten().astype(np.float32)
    
    print(f"📊 Data shapes:")
    print(f"   x_train: {x_train.shape} -> flat: {x_train_flat.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   x_test: {x_test.shape} -> flat: {x_test_flat.shape}")
    print(f"   y_test: {y_test.shape}")
    
    # Verification statistics
    print(f"\n🔍 Verification statistics:")
    print(f"   Train sample 1 sum: {x_train[0].sum():.6f}")
    print(f"   Train sample 1 mean: {x_train[0].mean():.6f}")
    print(f"   Test sample 1 sum: {x_test[0].sum():.6f}")
    print(f"   Test sample 1 mean: {x_test[0].mean():.6f}")
    print(f"   First 5 train labels: {y_train[:5]}")
    print(f"   First 5 test labels: {y_test[:5]}")
    
    # Export to binary files
    print(f"\n💾 Exporting binary files...")
    x_train_flat.tofile('cifar10_train_x_verified.bin')
    y_train.tofile('cifar10_train_y_verified.bin')
    x_test_flat.tofile('cifar10_test_x_verified.bin')
    y_test.tofile('cifar10_test_y_verified.bin')
    
    print(f"✅ Export complete!")
    print(f"\n📁 Created files:")
    print(f"   - cifar10_train_x_verified.bin ({x_train_flat.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   - cifar10_train_y_verified.bin ({y_train.nbytes / 1024:.1f} KB)")
    print(f"   - cifar10_test_x_verified.bin ({x_test_flat.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   - cifar10_test_y_verified.bin ({y_test.nbytes / 1024:.1f} KB)")

if __name__ == "__main__":
    export_cifar10_binary()