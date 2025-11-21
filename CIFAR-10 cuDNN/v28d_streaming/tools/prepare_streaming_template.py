#!/usr/bin/env python3
"""
Template: Prepare Dataset Directly in Streaming Format

This template shows how to save a dataset DIRECTLY in sample-major format
for streaming mode. Use this when your dataset is too large to fit in memory
twice (once for loading, once for converting).

Key principle: Write samples contiguously, one after another.
    [sample_0_features][sample_1_features][sample_2_features]...

Output files use _stream suffix to prevent format mixing:
    images_train_stream.bin
    labels_train_stream.bin
    images_test_stream.bin
    labels_test_stream.bin

Usage:
    1. Copy this template to your dataset directory
    2. Modify the load_and_process_sample() function for your data source
    3. Update N_SAMPLES, N_FEATURES, and output directory
    4. Run: python prepare_streaming.py

Author: v28d Streaming Architecture
Date: 2025-11-21
"""

import os
import struct
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

# =============================================================================
# CONFIGURATION - Modify these for your dataset
# =============================================================================

N_TRAIN_SAMPLES = 50000  # Number of training samples
N_TEST_SAMPLES = 10000  # Number of test samples
N_FEATURES = 3072  # Features per sample (e.g., 32x32x3 = 3072)
N_CLASSES = 10  # Number of classes

# Output directory (will be created if doesn't exist)
OUTPUT_DIR = "mydata_streaming"

# Data type for features (float32 for neural networks)
DTYPE = np.float32


# =============================================================================
# DATA LOADING - Modify this section for your data source
# =============================================================================


def load_and_process_sample(sample_idx: int, split: str) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess a single sample.

    This is where you implement your data loading logic.
    The function should return:
        - features: 1D numpy array of shape (N_FEATURES,), dtype=float32
        - label: integer class label

    Args:
        sample_idx: Index of the sample to load (0 to N_SAMPLES-1)
        split: 'train' or 'test'

    Returns:
        Tuple of (features_array, label)

    Examples of data sources you might load from:
        - Individual image files (JPG, PNG)
        - HDF5 datasets
        - Database queries
        - Network streams
        - Compressed archives
    """
    # =================================================================
    # EXAMPLE 1: Loading from individual image files
    # =================================================================
    # from PIL import Image
    #
    # if split == 'train':
    #     img_path = f'raw_data/train/image_{sample_idx:06d}.jpg'
    #     label_path = f'raw_data/train/label_{sample_idx:06d}.txt'
    # else:
    #     img_path = f'raw_data/test/image_{sample_idx:06d}.jpg'
    #     label_path = f'raw_data/test/label_{sample_idx:06d}.txt'
    #
    # # Load and preprocess image
    # img = Image.open(img_path).resize((32, 32))
    # img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    # features = img_array.flatten()  # Shape: (3072,) for 32x32x3
    #
    # # Load label
    # with open(label_path) as f:
    #     label = int(f.read().strip())
    #
    # return features, label

    # =================================================================
    # EXAMPLE 2: Loading from HDF5 file (memory-mapped, doesn't load all)
    # =================================================================
    # import h5py
    #
    # with h5py.File('large_dataset.h5', 'r') as f:
    #     if split == 'train':
    #         features = f['train_images'][sample_idx].astype(np.float32) / 255.0
    #         label = int(f['train_labels'][sample_idx])
    #     else:
    #         features = f['test_images'][sample_idx].astype(np.float32) / 255.0
    #         label = int(f['test_labels'][sample_idx])
    #
    # return features.flatten(), label

    # =================================================================
    # EXAMPLE 3: Generating synthetic data (for testing)
    # =================================================================
    np.random.seed(sample_idx + (0 if split == "train" else 1000000))
    label = sample_idx % N_CLASSES
    # Create features with some class-dependent pattern
    features = np.random.randn(N_FEATURES).astype(DTYPE) * 0.1
    features[
        label * (N_FEATURES // N_CLASSES) : (label + 1) * (N_FEATURES // N_CLASSES)
    ] += 0.5
    features = (features - features.min()) / (
        features.max() - features.min()
    )  # Normalize to [0,1]

    return features.astype(DTYPE), label


def sample_generator(split: str, n_samples: int) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Generator that yields samples one at a time.

    Override this if you have a more efficient way to iterate through your data.
    """
    for idx in range(n_samples):
        yield load_and_process_sample(idx, split)


# =============================================================================
# STREAMING FORMAT WRITER - Usually no need to modify below this line
# =============================================================================


def write_streaming_format(
    output_dir: str,
    split: str,
    n_samples: int,
    n_features: int,
    sample_generator_fn,
    progress_interval: int = 1000,
) -> dict:
    """
    Write samples directly to streaming format (sample-major binary).

    This writes samples one at a time, so memory usage is constant
    regardless of dataset size.

    Args:
        output_dir: Directory to write output files
        split: 'train' or 'test'
        n_samples: Total number of samples
        n_features: Features per sample
        sample_generator_fn: Function that yields (features, label) tuples
        progress_interval: Print progress every N samples

    Returns:
        Dictionary with statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Output file paths with _stream suffix
    data_path = os.path.join(output_dir, f"images_{split}_stream.bin")
    label_path = os.path.join(output_dir, f"labels_{split}_stream.bin")

    print(f"\nWriting {split} data in streaming format...")
    print(f"  Output: {data_path}")
    print(f"  Samples: {n_samples:,}")
    print(f"  Features: {n_features:,}")
    print(f"  Expected size: {n_samples * n_features * 4 / (1024**2):.1f} MB")

    stats = {
        "n_samples": 0,
        "n_features": n_features,
        "data_min": float("inf"),
        "data_max": float("-inf"),
        "data_sum": 0.0,
        "label_counts": {},
    }

    with open(data_path, "wb") as data_file, open(label_path, "wb") as label_file:
        for idx, (features, label) in enumerate(sample_generator_fn(split, n_samples)):
            # Validate features
            if len(features) != n_features:
                raise ValueError(
                    f"Sample {idx}: expected {n_features} features, got {len(features)}"
                )

            # Ensure correct dtype and write features
            features = features.astype(DTYPE)
            features.tofile(data_file)

            # Write label as int32
            label_file.write(struct.pack("i", int(label)))

            # Update statistics
            stats["n_samples"] += 1
            stats["data_min"] = min(stats["data_min"], features.min())
            stats["data_max"] = max(stats["data_max"], features.max())
            stats["data_sum"] += features.sum()
            stats["label_counts"][label] = stats["label_counts"].get(label, 0) + 1

            # Progress
            if (idx + 1) % progress_interval == 0 or idx == n_samples - 1:
                pct = 100 * (idx + 1) / n_samples
                print(f"  Progress: {idx + 1:,}/{n_samples:,} ({pct:.1f}%)")

    # Compute final statistics
    total_elements = stats["n_samples"] * n_features
    stats["data_mean"] = stats["data_sum"] / total_elements

    # Verify file sizes
    expected_data_size = n_samples * n_features * 4
    expected_label_size = n_samples * 4
    actual_data_size = os.path.getsize(data_path)
    actual_label_size = os.path.getsize(label_path)

    if actual_data_size != expected_data_size:
        raise RuntimeError(
            f"Data file size mismatch: {actual_data_size} != {expected_data_size}"
        )
    if actual_label_size != expected_label_size:
        raise RuntimeError(
            f"Label file size mismatch: {actual_label_size} != {expected_label_size}"
        )

    print(f"  Completed: {stats['n_samples']:,} samples")
    print(f"  Data range: [{stats['data_min']:.4f}, {stats['data_max']:.4f}]")
    print(f"  Data mean: {stats['data_mean']:.4f}")
    print(f"  Labels: {dict(sorted(stats['label_counts'].items()))}")

    return stats


def verify_streaming_format(
    output_dir: str, split: str, n_samples: int, n_features: int
) -> bool:
    """
    Verify the streaming format files are correct.

    Checks:
    1. File sizes match expected
    2. First and last samples can be read correctly
    3. Data is in valid range
    """
    data_path = os.path.join(output_dir, f"images_{split}_stream.bin")
    label_path = os.path.join(output_dir, f"labels_{split}_stream.bin")

    print(f"\nVerifying {split} data...")

    # Check file sizes
    expected_data_size = n_samples * n_features * 4
    expected_label_size = n_samples * 4

    actual_data_size = os.path.getsize(data_path)
    actual_label_size = os.path.getsize(label_path)

    if actual_data_size != expected_data_size:
        print(
            f"  FAIL: Data file size {actual_data_size} != expected {expected_data_size}"
        )
        return False
    print(f"  [OK] Data file size: {actual_data_size:,} bytes")

    if actual_label_size != expected_label_size:
        print(
            f"  FAIL: Label file size {actual_label_size} != expected {expected_label_size}"
        )
        return False
    print(f"  [OK] Label file size: {actual_label_size:,} bytes")

    # Read and check first sample
    data = np.memmap(data_path, dtype=DTYPE, mode="r")
    labels = np.memmap(label_path, dtype=np.int32, mode="r")

    first_sample = data[0:n_features]
    last_sample = data[(n_samples - 1) * n_features : n_samples * n_features]

    print(
        f"  [OK] First sample: min={first_sample.min():.4f}, max={first_sample.max():.4f}"
    )
    print(
        f"  [OK] Last sample: min={last_sample.min():.4f}, max={last_sample.max():.4f}"
    )
    print(f"  [OK] Label range: [{labels.min()}, {labels.max()}]")

    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("Preparing Dataset in Streaming Format")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Training samples: {N_TRAIN_SAMPLES:,}")
    print(f"  Test samples:     {N_TEST_SAMPLES:,}")
    print(f"  Features:         {N_FEATURES:,}")
    print(f"  Classes:          {N_CLASSES}")
    print(f"  Output:           {OUTPUT_DIR}/")
    print(f"  File suffix:      _stream.bin")

    # Write training data
    write_streaming_format(
        OUTPUT_DIR, "train", N_TRAIN_SAMPLES, N_FEATURES, sample_generator
    )

    # Write test data
    write_streaming_format(
        OUTPUT_DIR, "test", N_TEST_SAMPLES, N_FEATURES, sample_generator
    )

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    train_ok = verify_streaming_format(OUTPUT_DIR, "train", N_TRAIN_SAMPLES, N_FEATURES)
    test_ok = verify_streaming_format(OUTPUT_DIR, "test", N_TEST_SAMPLES, N_FEATURES)

    print("\n" + "=" * 60)
    if train_ok and test_ok:
        print("SUCCESS! Dataset ready for streaming mode.")
        print(f"\nOutput files:")
        for f in sorted(Path(OUTPUT_DIR).glob("*_stream.bin")):
            size_mb = f.stat().st_size / (1024**2)
            print(f"  {f.name:30s} {size_mb:10.2f} MB")
        print(f"\nTo use in training:")
        print(f"  1. Update your config to point to {OUTPUT_DIR}/")
        print(f"  2. Run with: ./train --stream")
    else:
        print("FAILED! Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
