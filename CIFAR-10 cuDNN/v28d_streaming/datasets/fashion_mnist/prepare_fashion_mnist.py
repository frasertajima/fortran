"""
Fashion-MNIST Preprocessing for v28 Baseline (Modular Framework)
==================================================================
Prepares Fashion-MNIST dataset in binary format for CUDA Fortran training.

Dataset: 60,000 training + 10,000 test grayscale 28x28 images, 10 classes

Classes: 
  0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
  5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot

Output format (matching CIFAR-10 pattern):
  - images_train.bin: (60000, 784) float32, normalized [0, 1]
  - labels_train.bin: (60000) int32, range [0, 9]
  - images_test.bin: (10000, 784) float32
  - labels_test.bin: (10000) int32
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

# Configuration
DATA_ROOT = './data'
OUTPUT_DIR = 'fashion_mnist_data'

if __name__ == '__main__':
    print("=" * 70)
    print("Fashion-MNIST Preprocessing for v28 Baseline (Modular Framework)")
    print("=" * 70)
    print()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Simple normalization to [0, 1] (ToTensor does this automatically)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to tensor and normalizes to [0, 1]
    ])

    # Download and load Fashion-MNIST
    print("Downloading/Loading Fashion-MNIST dataset...")
    trainset = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    print(f"  Training set: {len(trainset)} samples")
    print(f"  Test set:     {len(testset)} samples")
    print()

    # Convert to numpy arrays
    print("Converting to numpy arrays...")

    # Training data
    train_images = []
    train_labels = []
    for img, label in trainset:
        train_images.append(img.numpy().flatten())  # 28x28 -> 784
        train_labels.append(label)

    train_images = np.array(train_images, dtype=np.float32)  # (60000, 784)
    train_labels = np.array(train_labels, dtype=np.int32)    # (60000,)

    # Test data
    test_images = []
    test_labels = []
    for img, label in testset:
        test_images.append(img.numpy().flatten())  # 28x28 -> 784
        test_labels.append(label)

    test_images = np.array(test_images, dtype=np.float32)   # (10000, 784)
    test_labels = np.array(test_labels, dtype=np.int32)     # (10000,)

    print(f"  ✓ Training images: {train_images.shape} float32")
    print(f"  ✓ Training labels: {train_labels.shape} int32")
    print(f"  ✓ Test images:     {test_images.shape} float32")
    print(f"  ✓ Test labels:     {test_labels.shape} int32")
    print()

    # Verify data ranges
    print("Data validation:")
    print(f"  Image range: [{train_images.min():.4f}, {train_images.max():.4f}]")
    print(f"  Label range: [{train_labels.min()}, {train_labels.max()}]")
    print()

    # Transpose for Fortran column-major: (N, 784) → (784, N)
    # Python writes (784, N) in row-major → Fortran reads as (N, 784) in column-major ✓
    print("Transposing for Fortran column-major order...")
    train_images_fortran = train_images.T  # (784, 60000)
    test_images_fortran = test_images.T    # (784, 10000)
    print(f"  Train transposed: {train_images_fortran.shape} (784, N)")
    print(f"  Test transposed: {test_images_fortran.shape} (784, N)")
    print(f"  Fortran will read as: (N, 784)")
    print()

    # Save to binary files
    print("Saving binary files...")
    train_images_fortran.tofile(f'{OUTPUT_DIR}/images_train.bin')
    train_labels.tofile(f'{OUTPUT_DIR}/labels_train.bin')
    test_images_fortran.tofile(f'{OUTPUT_DIR}/images_test.bin')
    test_labels.tofile(f'{OUTPUT_DIR}/labels_test.bin')

    print(f"  ✓ {OUTPUT_DIR}/images_train.bin ({train_images.nbytes / 1024 / 1024:.2f} MB)")
    print(f"  ✓ {OUTPUT_DIR}/labels_train.bin ({train_labels.nbytes / 1024:.2f} KB)")
    print(f"  ✓ {OUTPUT_DIR}/images_test.bin ({test_images.nbytes / 1024 / 1024:.2f} MB)")
    print(f"  ✓ {OUTPUT_DIR}/labels_test.bin ({test_labels.nbytes / 1024:.2f} KB)")
    print()

    print("=" * 70)
    print("✅ Fashion-MNIST preprocessing complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  bash compile_fashion_mnist.sh && ./fashion_mnist_train")
    print()
