#!/usr/bin/env python3
"""
CIFAR-10 Data Loader and Preprocessor for CUDA Fortran (v27)

This script loads CIFAR-10 dataset, applies preprocessing, and saves
images/labels in Fortran-compatible binary format for ultra-fast training.

Follows the successful Oxford Flowers workflow pattern:
1. Python handles data loading and preprocessing (familiar, accessible)
2. CUDA Fortran handles training (fast, efficient)

Output files:
- cifar10_data/images_train.bin (50000 images, 32x32x3, transposed for Fortran)
- cifar10_data/labels_train.bin (50000 labels)
- cifar10_data/images_test.bin (10000 images, 32x32x3, transposed for Fortran)
- cifar10_data/labels_test.bin (10000 labels)

Data format:
- Images: float32, normalized [0, 1], shape saved as Fortran (3, 32, 32, N)
- Labels: int32, range [0, 9]
"""

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

# Configuration
DATA_DIR = Path("cifar-10-batches-py")
OUTPUT_DIR = Path("cifar10_data")
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_cifar10():
    """Download CIFAR-10 dataset if not present."""
    if DATA_DIR.exists():
        print(f"CIFAR-10 data already exists in {DATA_DIR}")
        return

    print("Downloading CIFAR-10 dataset...")
    tar_path = "cifar-10-python.tar.gz"

    if not Path(tar_path).exists():
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        print(f"Downloaded to {tar_path}")

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall()

    print(f"Extracted to {DATA_DIR}")


def load_batch(filename):
    """Load a single CIFAR-10 batch file."""
    with open(filename, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    # Extract images and labels
    images = batch[b"data"]  # (10000, 3072) uint8
    labels = batch[b"labels"]  # (10000,) int

    # Reshape images to (10000, 3, 32, 32)
    images = images.reshape(-1, 3, 32, 32)

    return images, np.array(labels)


def load_cifar10():
    """Load complete CIFAR-10 dataset (train + test)."""
    print("\nLoading CIFAR-10 dataset...")

    # Load training batches
    train_images = []
    train_labels = []

    for i in range(1, 6):
        batch_file = DATA_DIR / f"data_batch_{i}"
        images, labels = load_batch(batch_file)
        train_images.append(images)
        train_labels.append(labels)
        print(f"  Loaded training batch {i}/5: {images.shape}")

    # Concatenate training batches
    train_images = np.concatenate(train_images, axis=0)  # (50000, 3, 32, 32)
    train_labels = np.concatenate(train_labels, axis=0)  # (50000,)

    # Load test batch
    test_file = DATA_DIR / "test_batch"
    test_images, test_labels = load_batch(test_file)
    print(f"  Loaded test batch: {test_images.shape}")

    return train_images, train_labels, test_images, test_labels


def preprocess_images(images):
    """
    Preprocess images: uint8 [0, 255] → float32 [0, 1]

    Args:
        images: (N, 3, 32, 32) uint8 array

    Returns:
        (N, 3, 32, 32) float32 array, normalized to [0, 1]
    """
    # Convert to float32 and normalize to [0, 1]
    images = images.astype(np.float32) / 255.0
    return images


def save_fortran_format(output_dir):
    """
    Save CIFAR-10 in Fortran-compatible binary format.

    Fortran reads in column-major order, so we need to transpose carefully.
    For images (N, 3, 32, 32) in Python (row-major), we write the transpose
    so Fortran reads them correctly as (3, 32, 32, N).
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Download if needed
    download_cifar10()

    # Load dataset
    train_images, train_labels, test_images, test_labels = load_cifar10()

    print(f"\nDataset shapes:")
    print(f"  Train images: {train_images.shape} (N, C, H, W)")
    print(f"  Train labels: {train_labels.shape}")
    print(f"  Test images: {test_images.shape}")
    print(f"  Test labels: {test_labels.shape}")

    # Preprocess
    print("\nPreprocessing...")
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    print(f"  Train range: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"  Test range: [{test_images.min():.3f}, {test_images.max():.3f}]")

    # Flatten images to 2D (simpler for Fortran)
    print("\nFlattening images to (N, 3072)...")
    train_flat = train_images.reshape(train_images.shape[0], -1)  # (50000, 3072)
    test_flat = test_images.reshape(test_images.shape[0], -1)  # (10000, 3072)

    print(f"  Train flattened: {train_flat.shape}")
    print(f"  Test flattened: {test_flat.shape}")

    # Transpose for Fortran column-major: (N, 3072) → (3072, N)
    # Python writes (3072, N) in row-major → Fortran reads as (N, 3072) in column-major ✓
    print("\nTransposing for Fortran column-major order...")
    train_flat_fortran = train_flat.T  # (3072, 50000)
    test_flat_fortran = test_flat.T  # (3072, 10000)

    print(f"  Train transposed: {train_flat_fortran.shape} (3072, N)")
    print(f"  Test transposed: {test_flat_fortran.shape} (3072, N)")
    print(f"  Fortran will read as: (N, 3072)")

    # Save training data
    print("\nSaving binary files (Fortran-compatible format)...")
    train_images_path = output_dir / "images_train.bin"
    train_labels_path = output_dir / "labels_train.bin"

    train_flat_fortran.tofile(train_images_path)
    train_labels.astype(np.int32).tofile(train_labels_path)
    print(f"  ✓ {train_images_path} [{train_flat_fortran.shape}]")
    print(f"  ✓ {train_labels_path} [{train_labels.shape}]")

    # Save test data
    test_images_path = output_dir / "images_test.bin"
    test_labels_path = output_dir / "labels_test.bin"

    test_flat_fortran.tofile(test_images_path)
    test_labels.astype(np.int32).tofile(test_labels_path)
    print(f"  ✓ {test_images_path} [{test_flat_fortran.shape}]")
    print(f"  ✓ {test_labels_path} [{test_labels.shape}]")

    # Verification
    print("\nVerification - First image, first pixel:")
    print(f"  Train image[0, :, 0, 0] (RGB): {train_images[0, :, 0, 0]}")
    print(f"  Test image[0, :, 0, 0] (RGB): {test_images[0, :, 0, 0]}")

    # File sizes
    print("\nFile sizes:")
    for path in [
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
    ]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name}: {size_mb:.2f} MB")

    print("\n✅ CIFAR-10 data ready for CUDA Fortran training!")
    print(f"   Files saved to: {output_dir}/")


if __name__ == "__main__":
    print("=" * 70)
    print("CIFAR-10 Data Preparation for CUDA Fortran (v27)")
    print("=" * 70)

    save_fortran_format(OUTPUT_DIR)

    print("\nNext step:")
    print("  Compile and run: bash ./compile_cifar10_v27.sh && ./cifar10_cudnn_v27")
