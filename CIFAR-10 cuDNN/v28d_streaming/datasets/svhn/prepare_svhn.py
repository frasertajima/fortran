#!/usr/bin/env python3
"""
SVHN (Street View House Numbers) Data Loader and Preprocessor for CUDA Fortran (v28 Baseline)

This script loads the SVHN dataset, applies preprocessing, and saves
images/labels in Fortran-compatible binary format for ultra-fast CNN training.

Follows the modular v28 baseline workflow pattern:
1. Python handles data loading and preprocessing (familiar, accessible)
2. CUDA Fortran handles training (fast, efficient, modular)

Output files:
- svhn_data/images_train.bin (73257 images, 32x32x3, flattened & transposed)
- svhn_data/labels_train.bin (73257 labels)
- svhn_data/images_test.bin (26032 images, 32x32x3, flattened & transposed)
- svhn_data/labels_test.bin (26032 labels)

Data format:
- Images: float32, normalized [0, 1], saved as (3072, N) for Fortran
- Labels: int32, range [0, 9] (digit recognition)
"""

import numpy as np
from pathlib import Path
import urllib.request

# Configuration
OUTPUT_DIR = Path('svhn_data')
SVHN_TRAIN_URL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
SVHN_TEST_URL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

def download_svhn():
    """Download SVHN dataset if not present."""
    train_path = Path('train_32x32.mat')
    test_path = Path('test_32x32.mat')

    if not train_path.exists():
        print(f'Downloading SVHN training set...')
        urllib.request.urlretrieve(SVHN_TRAIN_URL, train_path)
        print(f'  ✓ Downloaded {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)')
    else:
        print(f'SVHN training set already exists: {train_path}')

    if not test_path.exists():
        print(f'Downloading SVHN test set...')
        urllib.request.urlretrieve(SVHN_TEST_URL, test_path)
        print(f'  ✓ Downloaded {test_path} ({test_path.stat().st_size / 1e6:.1f} MB)')
    else:
        print(f'SVHN test set already exists: {test_path}')

    return train_path, test_path

def load_svhn():
    """
    Load SVHN dataset from .mat files.

    Returns:
        train_images: (N, 3, 32, 32) float32 array
        train_labels: (N,) int32 array
        test_images: (N, 3, 32, 32) float32 array
        test_labels: (N,) int32 array
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        print("ERROR: scipy is required. Install with: pip install scipy")
        raise

    print('\nLoading SVHN dataset...')

    # Download if needed
    train_path, test_path = download_svhn()

    # Load training data
    print(f'  Loading {train_path}...')
    train_data = loadmat(str(train_path))
    train_images = train_data['X']  # (32, 32, 3, N) uint8
    train_labels = train_data['y'].flatten()  # (N,) - labels are 1-10, where 10 = digit 0

    # Load test data
    print(f'  Loading {test_path}...')
    test_data = loadmat(str(test_path))
    test_images = test_data['X']  # (32, 32, 3, N) uint8
    test_labels = test_data['y'].flatten()  # (N,)

    print(f'\nOriginal MATLAB format:')
    print(f'  Train images: {train_images.shape} (H, W, C, N)')
    print(f'  Train labels: {train_labels.shape}')
    print(f'  Test images: {test_images.shape} (H, W, C, N)')
    print(f'  Test labels: {test_labels.shape}')

    # Transpose to (N, C, H, W) - standard PyTorch/CNN format
    train_images = np.transpose(train_images, (3, 2, 0, 1))  # (N, C, H, W)
    test_images = np.transpose(test_images, (3, 2, 0, 1))    # (N, C, H, W)

    # Convert labels: SVHN uses 1-10 where 10 = digit 0
    # We need 0-9 where index matches the digit
    train_labels = np.where(train_labels == 10, 0, train_labels)
    test_labels = np.where(test_labels == 10, 0, test_labels)

    print(f'\nTransposed to CNN format:')
    print(f'  Train images: {train_images.shape} (N, C, H, W)')
    print(f'  Train labels: {train_labels.shape}, range [{train_labels.min()}, {train_labels.max()}]')
    print(f'  Test images: {test_images.shape} (N, C, H, W)')
    print(f'  Test labels: {test_labels.shape}, range [{test_labels.min()}, {test_labels.max()}]')

    return train_images, train_labels, test_images, test_labels

def preprocess_images(images):
    """
    Preprocess images: uint8 [0, 255] → float32 [0, 1]

    Args:
        images: (N, 3, 32, 32) uint8 array

    Returns:
        (N, 3, 32, 32) float32 array, normalized to [0, 1]
    """
    images = images.astype(np.float32) / 255.0
    return images

def save_fortran_format(output_dir):
    """
    Save SVHN in Fortran-compatible binary format.

    Fortran reads in column-major order, so we:
    1. Flatten images: (N, 3, 32, 32) → (N, 3072)
    2. Transpose: (N, 3072) → (3072, N)
    3. Write transposed data
    4. Fortran reads as (N, 3072) in column-major ✓
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    train_images, train_labels, test_images, test_labels = load_svhn()

    # Preprocess
    print('\nPreprocessing...')
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    print(f'  Train range: [{train_images.min():.3f}, {train_images.max():.3f}]')
    print(f'  Test range: [{test_images.min():.3f}, {test_images.max():.3f}]')

    # Flatten images to 2D (simpler for Fortran)
    print('\nFlattening images to (N, 3072)...')
    train_flat = train_images.reshape(train_images.shape[0], -1)  # (73257, 3072)
    test_flat = test_images.reshape(test_images.shape[0], -1)      # (26032, 3072)

    print(f'  Train flattened: {train_flat.shape}')
    print(f'  Test flattened: {test_flat.shape}')

    # Transpose for Fortran column-major: (N, 3072) → (3072, N)
    # Python writes (3072, N) in row-major → Fortran reads as (N, 3072) in column-major ✓
    print('\nTransposing for Fortran column-major order...')
    train_flat_fortran = train_flat.T  # (3072, 73257)
    test_flat_fortran = test_flat.T    # (3072, 26032)

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
    print(f'  Train label[0]: {train_labels[0]}')
    print(f'  Test image[0, :, 0, 0] (RGB): {test_images[0, :, 0, 0]}')
    print(f'  Test label[0]: {test_labels[0]}')

    # File sizes
    print('\nFile sizes:')
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f'  {path.name}: {size_mb:.2f} MB')

    print('\n✅ SVHN data ready for CUDA Fortran training!')
    print(f'   Files saved to: {output_dir}/')
    print(f'\n   Train samples: {train_flat.shape[0]:,}')
    print(f'   Test samples:  {test_flat.shape[0]:,}')

if __name__ == '__main__':
    print('=' * 70)
    print('SVHN Data Preparation for v28 Baseline (Modular Framework)')
    print('=' * 70)

    save_fortran_format(OUTPUT_DIR)

    print('\nNext step:')
    print('  Compile and run: bash compile_svhn.sh && ./svhn_train')
