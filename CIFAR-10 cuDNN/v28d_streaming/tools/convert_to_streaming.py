#!/usr/bin/env python3
"""
Convert dataset binary files from feature-major to sample-major format for streaming.

The streaming data loader requires sample-major layout where each sample's features
are stored contiguously. This script converts feature-major files to sample-major
format with a _stream suffix to prevent accidental format mixing.

Usage:
    # Convert single file:
    python convert_to_streaming.py --input cifar10_data/images_train.bin \
                                   --samples 50000 --features 3072

    # Convert entire directory with preset:
    python convert_to_streaming.py --input-dir cifar10_data/ \
                                   --output-dir cifar10_data_streaming/ \
                                   --preset cifar10

Author: Claude Code Assistant
Date: 2025-11-21
"""

import argparse
import os
from pathlib import Path

import numpy as np


def convert_feature_to_sample_major(
    input_path: str,
    output_path: str,
    n_samples: int,
    n_features: int,
    verify: bool = True,
) -> dict:
    """
    Convert a feature-major binary file to sample-major format.

    Feature-major layout: [f0_s0, f0_s1, ..., f0_sN, f1_s0, f1_s1, ..., f1_sN, ...]
    Sample-major layout:  [s0_f0, s0_f1, ..., s0_fM, s1_f0, s1_f1, ..., s1_fM, ...]

    Returns dict with conversion statistics.
    """
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "n_samples": n_samples,
        "n_features": n_features,
        "success": False,
    }

    # Load data
    print(f"Loading {input_path}...")
    data = np.fromfile(input_path, dtype=np.float32)

    expected_size = n_samples * n_features
    if len(data) != expected_size:
        raise ValueError(
            f"File size mismatch: got {len(data)} floats, expected {expected_size} "
            f"({n_samples} samples x {n_features} features)"
        )

    stats["file_size_mb"] = len(data) * 4 / (1024 * 1024)
    stats["data_min"] = float(data.min())
    stats["data_max"] = float(data.max())
    stats["data_mean"] = float(data.mean())

    # Reshape as feature-major: (features, samples)
    print(f"Reshaping from feature-major ({n_features}, {n_samples})...")
    data_fm = data.reshape(n_features, n_samples)

    # Transpose to sample-major: (samples, features)
    print(f"Transposing to sample-major ({n_samples}, {n_features})...")
    data_sm = data_fm.T

    # Ensure C-contiguous for correct binary output
    data_sm = np.ascontiguousarray(data_sm, dtype=np.float32)

    # Save
    print(f"Writing to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data_sm.tofile(output_path)

    # Verify conversion
    if verify:
        print("Verifying conversion...")
        data_verify = np.fromfile(output_path, dtype=np.float32)

        # Sample 0 in sample-major should be first n_features values
        sample_0_sm = data_verify[0:n_features]

        # Sample 0 in feature-major is at positions 0, n_samples, 2*n_samples, ...
        sample_0_fm = data[0::n_samples][:n_features]

        if np.allclose(sample_0_sm, sample_0_fm):
            print("  [OK] Sample 0 verification PASSED")
            stats["verify_sample_0"] = True
        else:
            print("  [FAIL] Sample 0 verification FAILED")
            stats["verify_sample_0"] = False

        # Check a middle sample
        mid_idx = n_samples // 2
        sample_mid_sm = data_verify[mid_idx * n_features : (mid_idx + 1) * n_features]
        sample_mid_fm = data[mid_idx::n_samples][:n_features]

        if np.allclose(sample_mid_sm, sample_mid_fm):
            print(f"  [OK] Sample {mid_idx} verification PASSED")
            stats["verify_sample_mid"] = True
        else:
            print(f"  [FAIL] Sample {mid_idx} verification FAILED")
            stats["verify_sample_mid"] = False

        # Check last sample
        sample_last_sm = data_verify[
            (n_samples - 1) * n_features : n_samples * n_features
        ]
        sample_last_fm = data[n_samples - 1 :: n_samples][:n_features]

        if np.allclose(sample_last_sm, sample_last_fm):
            print(f"  [OK] Sample {n_samples - 1} verification PASSED")
            stats["verify_sample_last"] = True
        else:
            print(f"  [FAIL] Sample {n_samples - 1} verification FAILED")
            stats["verify_sample_last"] = False

    stats["success"] = True
    return stats


def copy_labels(input_path: str, output_path: str) -> dict:
    """Copy label file (labels don't need conversion, just copy with new name)."""
    print(f"Copying labels: {input_path} -> {output_path}")

    data = np.fromfile(input_path, dtype=np.int32)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data.tofile(output_path)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "n_labels": len(data),
        "unique_labels": len(np.unique(data)),
    }


def convert_directory(
    input_dir: str,
    output_dir: str,
    n_samples_train: int,
    n_samples_test: int,
    n_features: int,
) -> None:
    """
    Convert all files in a dataset directory.

    Input files (feature-major):
    - images_train.bin
    - images_test.bin
    - labels_train.bin
    - labels_test.bin

    Output files (sample-major, with _stream suffix):
    - images_train_stream.bin
    - images_test_stream.bin
    - labels_train_stream.bin
    - labels_test_stream.bin
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print(f"\n{'=' * 60}")
    print(f"Converting dataset directory")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Files will have _stream suffix to prevent format mixing")
    print(f"{'=' * 60}\n")

    # Training images
    train_img = input_path / "images_train.bin"
    if train_img.exists():
        print("\n--- Training Images ---")
        convert_feature_to_sample_major(
            str(train_img),
            str(output_path / "images_train_stream.bin"),
            n_samples_train,
            n_features,
        )
    else:
        print(f"Warning: {train_img} not found, skipping")

    # Test images
    test_img = input_path / "images_test.bin"
    if test_img.exists():
        print("\n--- Test Images ---")
        convert_feature_to_sample_major(
            str(test_img),
            str(output_path / "images_test_stream.bin"),
            n_samples_test,
            n_features,
        )
    else:
        print(f"Warning: {test_img} not found, skipping")

    # Training labels
    train_lbl = input_path / "labels_train.bin"
    if train_lbl.exists():
        print("\n--- Training Labels ---")
        copy_labels(str(train_lbl), str(output_path / "labels_train_stream.bin"))
    else:
        print(f"Warning: {train_lbl} not found, skipping")

    # Test labels
    test_lbl = input_path / "labels_test.bin"
    if test_lbl.exists():
        print("\n--- Test Labels ---")
        copy_labels(str(test_lbl), str(output_path / "labels_test_stream.bin"))
    else:
        print(f"Warning: {test_lbl} not found, skipping")

    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"")
    print(f"Output files (sample-major format):")
    for f in sorted(output_path.glob("*_stream.bin")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s} {size_mb:8.2f} MB")
    print(f"")
    print(f"These files are ready for streaming mode.")
    print(f"{'=' * 60}\n")


# Preset configurations for common datasets
DATASET_PRESETS = {
    "cifar10": {
        "n_samples_train": 50000,
        "n_samples_test": 10000,
        "n_features": 3072,  # 32x32x3
    },
    "cifar100": {
        "n_samples_train": 50000,
        "n_samples_test": 10000,
        "n_features": 3072,  # 32x32x3
    },
    "fashion-mnist": {
        "n_samples_train": 60000,
        "n_samples_test": 10000,
        "n_features": 784,  # 28x28
    },
    "mnist": {
        "n_samples_train": 60000,
        "n_samples_test": 10000,
        "n_features": 784,  # 28x28
    },
    "svhn": {
        "n_samples_train": 73257,
        "n_samples_test": 26032,
        "n_features": 3072,  # 32x32x3
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset files from feature-major to sample-major format for streaming.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file conversion (output gets _stream suffix automatically):
  python convert_to_streaming.py --input images_train.bin --samples 50000 --features 3072

  # Directory conversion with preset:
  python convert_to_streaming.py --input-dir cifar10_data/ --output-dir cifar10_data_streaming/ \\
                                 --preset cifar10

  # Directory conversion with custom dimensions:
  python convert_to_streaming.py --input-dir mydata/ --output-dir mydata_streaming/ \\
                                 --samples-train 100000 --samples-test 10000 --features 2048

Available presets: cifar10, cifar100, fashion-mnist, mnist, svhn

Output files will have '_stream' suffix to prevent accidental format mixing.
        """,
    )

    # Input/output options
    parser.add_argument("--input", "-i", help="Input binary file path")
    parser.add_argument(
        "--output", "-o", help="Output binary file path (default: input_stream.bin)"
    )
    parser.add_argument("--input-dir", help="Input directory containing dataset files")
    parser.add_argument("--output-dir", help="Output directory for converted files")

    # Dimension options
    parser.add_argument(
        "--samples", "-n", type=int, help="Number of samples (for single file)"
    )
    parser.add_argument(
        "--features", "-f", type=int, help="Number of features per sample"
    )
    parser.add_argument(
        "--samples-train", type=int, help="Number of training samples (for directory)"
    )
    parser.add_argument(
        "--samples-test", type=int, help="Number of test samples (for directory)"
    )

    # Preset option
    parser.add_argument(
        "--preset",
        "-p",
        choices=list(DATASET_PRESETS.keys()),
        help="Use preset dimensions for known dataset",
    )

    # Flags
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip verification step"
    )
    parser.add_argument(
        "--list-presets", action="store_true", help="List available presets and exit"
    )

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("\nAvailable dataset presets:")
        print("-" * 50)
        for name, config in DATASET_PRESETS.items():
            print(f"  {name}:")
            print(f"    Training samples: {config['n_samples_train']:,}")
            print(f"    Test samples:     {config['n_samples_test']:,}")
            print(f"    Features:         {config['n_features']:,}")
            print()
        return

    # Load preset if specified
    if args.preset:
        preset = DATASET_PRESETS[args.preset]
        if args.samples_train is None:
            args.samples_train = preset["n_samples_train"]
        if args.samples_test is None:
            args.samples_test = preset["n_samples_test"]
        if args.features is None:
            args.features = preset["n_features"]
        if args.samples is None:
            args.samples = preset["n_samples_train"]

    # Directory mode
    if args.input_dir and args.output_dir:
        if not args.samples_train or not args.samples_test or not args.features:
            parser.error(
                "Directory mode requires --samples-train, --samples-test, and --features "
                "(or use --preset)"
            )
        convert_directory(
            args.input_dir,
            args.output_dir,
            args.samples_train,
            args.samples_test,
            args.features,
        )
        return

    # Single file mode
    if args.input:
        if not args.samples:
            parser.error("Single file mode requires --samples")

        # Auto-detect features if not provided
        if not args.features:
            file_size = os.path.getsize(args.input)
            args.features = file_size // (args.samples * 4)
            print(f"Auto-detected features: {args.features}")

        # Generate output path with _stream suffix if not provided
        if not args.output:
            input_path = Path(args.input)
            args.output = str(
                input_path.parent / f"{input_path.stem}_stream{input_path.suffix}"
            )
            print(f"Output file: {args.output}")

        stats = convert_feature_to_sample_major(
            args.input,
            args.output,
            args.samples,
            args.features,
            verify=not args.no_verify,
        )

        print(f"\nConversion statistics:")
        print(f"  File size: {stats['file_size_mb']:.2f} MB")
        print(f"  Data range: [{stats['data_min']:.4f}, {stats['data_max']:.4f}]")
        print(f"  Data mean: {stats['data_mean']:.4f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
