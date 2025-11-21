#!/usr/bin/env python3
"""
Generate Large Synthetic CIFAR-10-like Dataset for Managed Memory Testing

Creates a dataset sized to test managed memory capabilities:
- ~1 million training samples (~12GB)
- 10,000 test samples (same as CIFAR-10)
- Random data with realistic patterns
- 10 classes (0-9)
"""

import numpy as np
import os

def generate_large_dataset(n_train=1_000_000, n_test=10_000, seed=42):
    """
    Generate large synthetic dataset

    Args:
        n_train: Number of training samples (default 1M ≈ 12GB)
        n_test: Number of test samples
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    print("="*80)
    print("Generating Large Synthetic Dataset for Managed Memory Testing")
    print("="*80)
    print(f"  Training samples: {n_train:,}")
    print(f"  Test samples:     {n_test:,}")
    print(f"  Image size:       32x32x3")
    print(f"  Classes:          10")
    print("")

    # Calculate expected sizes
    train_size_gb = (n_train * 3072 * 4) / (1024**3)
    test_size_gb = (n_test * 3072 * 4) / (1024**3)
    total_size_gb = train_size_gb + test_size_gb

    print(f"Expected dataset size:")
    print(f"  Training images:  {train_size_gb:.2f} GB")
    print(f"  Test images:      {test_size_gb:.2f} GB")
    print(f"  Total:            {total_size_gb:.2f} GB")
    print("")

    # Create output directory
    os.makedirs('cifar10_data', exist_ok=True)

    # Generate training data in chunks to manage memory
    chunk_size = 50_000  # Process 50K samples at a time
    n_chunks = (n_train + chunk_size - 1) // chunk_size

    print(f"Generating training data ({n_chunks} chunks of {chunk_size:,} samples)...")

    # Pre-create train_images file
    train_images_path = 'cifar10_data/images_train.bin'
    train_labels_path = 'cifar10_data/labels_train.bin'

    # Generate training data in chunks
    train_labels_all = []

    with open(train_images_path, 'wb') as f_img:
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_train)
            actual_chunk_size = end_idx - start_idx

            if chunk_idx % 5 == 0:
                print(f"  Chunk {chunk_idx+1}/{n_chunks}: samples {start_idx:,}-{end_idx:,}")

            # Generate synthetic images with some structure
            # Mix of noise and simple patterns
            chunk_images = np.random.rand(actual_chunk_size, 3, 32, 32).astype(np.float32)

            # Add some simple patterns (stripes, gradients, etc.)
            for i in range(actual_chunk_size):
                class_id = i % 10
                # Add class-specific patterns
                if class_id < 5:
                    # Vertical stripes
                    chunk_images[i, :, :, ::2] *= 1.5
                else:
                    # Horizontal stripes
                    chunk_images[i, :, ::2, :] *= 1.5

            # Clip to [0, 1]
            chunk_images = np.clip(chunk_images, 0.0, 1.0)

            # Flatten: (N, 3, 32, 32) -> (N, 3072)
            chunk_images_flat = chunk_images.reshape(actual_chunk_size, -1)

            # Write to file (Fortran will read as transposed)
            chunk_images_flat.T.astype(np.float32).tofile(f_img)

            # Generate labels
            chunk_labels = np.array([i % 10 for i in range(start_idx, end_idx)], dtype=np.int32)
            train_labels_all.extend(chunk_labels)

    # Write all training labels
    train_labels_array = np.array(train_labels_all, dtype=np.int32)
    train_labels_array.tofile(train_labels_path)

    print(f"  Training data complete: {n_train:,} samples")
    print("")

    # Generate test data (small enough to do in one go)
    print(f"Generating test data ({n_test:,} samples)...")

    test_images = np.random.rand(n_test, 3, 32, 32).astype(np.float32)

    # Add similar patterns
    for i in range(n_test):
        class_id = i % 10
        if class_id < 5:
            test_images[i, :, :, ::2] *= 1.5
        else:
            test_images[i, :, ::2, :] *= 1.5

    test_images = np.clip(test_images, 0.0, 1.0)

    # Flatten and save
    test_images_flat = test_images.reshape(n_test, -1)
    test_images_flat.T.astype(np.float32).tofile('cifar10_data/images_test.bin')

    # Test labels
    test_labels = np.array([i % 10 for i in range(n_test)], dtype=np.int32)
    test_labels.tofile('cifar10_data/labels_test.bin')

    print(f"  Test data complete: {n_test:,} samples")
    print("")

    # Verify file sizes
    print("Verifying generated files...")
    for filename in ['images_train.bin', 'labels_train.bin', 'images_test.bin', 'labels_test.bin']:
        filepath = os.path.join('cifar10_data', filename)
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"  {filename:25s}: {size_mb:8.2f} MB")

    print("")
    print("="*80)
    print("✅ Dataset generation complete!")
    print("="*80)
    print("")
    print("Next steps:")
    print("  1. Compile: bash compile_cifar10.sh")
    print("  2. Train:   ./cifar10_train")
    print("  3. Monitor: Watch for managed memory paging with nvidia-smi")
    print("")
    print("Expected behavior:")
    print(f"  - Dataset size ({total_size_gb:.1f} GB) may exceed GPU RAM")
    print("  - Managed memory will automatically page data between CPU/GPU")
    print("  - Training should complete without OOM errors")
    print("="*80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate large synthetic dataset')
    parser.add_argument('--train', type=int, default=1_000_000,
                        help='Number of training samples (default: 1M ≈ 12GB)')
    parser.add_argument('--test', type=int, default=10_000,
                        help='Number of test samples (default: 10K)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    generate_large_dataset(n_train=args.train, n_test=args.test, seed=args.seed)
