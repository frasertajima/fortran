#!/usr/bin/env python3
"""
CIFAR-100 Data Loader and Preprocessor for CUDA Fortran (v28 Baseline)

This script loads CIFAR-100 dataset, applies preprocessing, and saves
images/labels in Fortran-compatible binary format for ultra-fast training.

Follows the modular v28 baseline workflow pattern:
1. Python handles data loading and preprocessing (familiar, accessible)
2. CUDA Fortran handles training (fast, efficient, modular)

Output files:
- cifar100_data/images_train.bin (50000 images, 32x32x3, transposed for Fortran)
- cifar100_data/labels_train.bin (50000 labels, fine labels 0-99)
- cifar100_data/images_test.bin (10000 images, 32x32x3, transposed for Fortran)
- cifar100_data/labels_test.bin (10000 labels, fine labels 0-99)

Data format:
- Images: float32, normalized [0, 1], shape saved as (3072, N) for Fortran
- Labels: int32, range [0, 99] (100 fine-grained classes)
"""

import numpy as np
import pickle
from pathlib import Path
import tarfile
import urllib.request

# Configuration
DATA_DIR = Path('cifar-100-python')
OUTPUT_DIR = Path('cifar100_data')
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

def download_cifar100():
    """Download CIFAR-100 dataset if not present."""
    if DATA_DIR.exists():
        print(f'CIFAR-100 data already exists in {DATA_DIR}')
        return

    print('Downloading CIFAR-100 dataset...')
    tar_path = 'cifar-100-python.tar.gz'

    if not Path(tar_path).exists():
        print(f'  Downloading from {CIFAR100_URL}...')
        urllib.request.urlretrieve(CIFAR100_URL, tar_path)
        print(f'  Downloaded to {tar_path}')

    print('Extracting...')
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall()

    print(f'Extracted to {DATA_DIR}')

def load_cifar100_file(filename):
    """
    Load a CIFAR-100 file.

    CIFAR-100 format:
    - 'data': (N, 3072) uint8 array
    - 'fine_labels': (N,) list - 100 fine-grained classes (0-99)
    - 'coarse_labels': (N,) list - 20 superclasses (0-19)
    """
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')

    # Extract images and fine labels
    images = data_dict[b'data']  # (N, 3072) uint8
    fine_labels = data_dict[b'fine_labels']  # (N,) - we use fine labels (100 classes)

    # Reshape images to (N, 3, 32, 32)
    images = images.reshape(-1, 3, 32, 32)

    return images, np.array(fine_labels)

def load_cifar100():
    """Load complete CIFAR-100 dataset (train + test)."""
    print('\nLoading CIFAR-100 dataset...')

    # Load training data
    train_file = DATA_DIR / 'train'
    print(f'  Loading training data from {train_file}...')
    train_images, train_labels = load_cifar100_file(train_file)
    print(f'  Loaded training: {train_images.shape}')

    # Load test data
    test_file = DATA_DIR / 'test'
    print(f'  Loading test data from {test_file}...')
    test_images, test_labels = load_cifar100_file(test_file)
    print(f'  Loaded test: {test_images.shape}')

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
    Save CIFAR-100 in Fortran-compatible binary format.

    Fortran reads in column-major order, so we:
    1. Flatten images: (N, 3, 32, 32) → (N, 3072)
    2. Transpose: (N, 3072) → (3072, N)
    3. Write transposed data
    4. Fortran reads as (N, 3072) in column-major ✓
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Download if needed
    download_cifar100()

    # Load dataset
    train_images, train_labels, test_images, test_labels = load_cifar100()

    print(f'\nDataset shapes:')
    print(f'  Train images: {train_images.shape} (N, C, H, W)')
    print(f'  Train labels: {train_labels.shape} - range [{train_labels.min()}, {train_labels.max()}]')
    print(f'  Test images: {test_images.shape}')
    print(f'  Test labels: {test_labels.shape} - range [{test_labels.min()}, {test_labels.max()}]')

    # Preprocess
    print('\nPreprocessing...')
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    print(f'  Train range: [{train_images.min():.3f}, {train_images.max():.3f}]')
    print(f'  Test range: [{test_images.min():.3f}, {test_images.max():.3f}]')

    # Flatten images to 2D (simpler for Fortran)
    print('\nFlattening images to (N, 3072)...')
    train_flat = train_images.reshape(train_images.shape[0], -1)  # (50000, 3072)
    test_flat = test_images.reshape(test_images.shape[0], -1)      # (10000, 3072)

    print(f'  Train flattened: {train_flat.shape}')
    print(f'  Test flattened: {test_flat.shape}')

    # Transpose for Fortran column-major: (N, 3072) → (3072, N)
    # Python writes (3072, N) in row-major → Fortran reads as (N, 3072) in column-major ✓
    print('\nTransposing for Fortran column-major order...')
    train_flat_fortran = train_flat.T  # (3072, 50000)
    test_flat_fortran = test_flat.T    # (3072, 10000)

    print(f'  Train transposed: {train_flat_fortran.shape} (3072, N)')
    print(f'  Test transposed: {test_flat_fortran.shape} (3072, N)')
    print(f'  Fortran will read as: (N, 3072)')

    # Save training data
    print('\nSaving binary files (Fortran-compatible format)...')
    train_images_path = output_dir / 'images_train.bin'
    train_labels_path = output_dir / 'labels_train.bin'

    train_flat_fortran.tofile(train_images_path)
    train_labels.astype(np.int32).tofile(train_labels_path)
    print(f'  ✓ {train_images_path} [{train_flat_fortran.shape}]')
    print(f'  ✓ {train_labels_path} [{train_labels.shape}]')

    # Save test data
    test_images_path = output_dir / 'images_test.bin'
    test_labels_path = output_dir / 'labels_test.bin'

    test_flat_fortran.tofile(test_images_path)
    test_labels.astype(np.int32).tofile(test_labels_path)
    print(f'  ✓ {test_images_path} [{test_flat_fortran.shape}]')
    print(f'  ✓ {test_labels_path} [{test_labels.shape}]')

    # Verification
    print('\nVerification - First image, first pixel:')
    print(f'  Train image[0, :, 0, 0] (RGB): {train_images[0, :, 0, 0]}')
    print(f'  Train label[0]: {train_labels[0]} (class 0-99)')
    print(f'  Test image[0, :, 0, 0] (RGB): {test_images[0, :, 0, 0]}')
    print(f'  Test label[0]: {test_labels[0]} (class 0-99)')

    # File sizes
    print('\nFile sizes:')
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f'  {path.name}: {size_mb:.2f} MB')

    print('\n✅ CIFAR-100 data ready for CUDA Fortran training!')
    print(f'   Files saved to: {output_dir}/')
    print(f'\n   100 fine-grained classes (0-99)')
    print(f'   Train samples: {train_flat.shape[0]:,}')
    print(f'   Test samples:  {test_flat.shape[0]:,}')

if __name__ == '__main__':
    print('=' * 70)
    print('CIFAR-100 Data Preparation for v28 Baseline (Modular Framework)')
    print('=' * 70)

    save_fortran_format(OUTPUT_DIR)

    print('\nNext step:')
    print('  Compile and run: bash compile_cifar100.sh && ./cifar100_train')
