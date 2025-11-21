"""
Test if PyTorch conv1 output matches what Fortran would produce.

This will tell us if the conv weight loading is correct.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add inference directory to path
sys.path.append("inference")
from model_loader import load_v28_model


def test_conv1_only():
    """Test just the first convolution layer"""

    MODEL_DIR = Path("v28_baseline/datasets/cifar10/saved_models/cifar10/")

    # Load test image that Fortran saw
    test_images = np.fromfile(
        "v28_baseline/datasets/cifar10/cifar10_data/images_test.bin", dtype=np.float32
    )
    test_images = test_images.reshape((3072, 10000), order="C")
    first_image = test_images[:, 0].reshape((3, 32, 32), order="C")  # (C, H, W)

    print("=" * 70)
    print("Testing Conv1 Output Match")
    print("=" * 70)

    # Load model
    model = load_v28_model(
        MODEL_DIR, in_channels=3, num_classes=10, input_size=32, device="cpu"
    )

    # Prepare input
    input_tensor = torch.from_numpy(first_image).unsqueeze(0)  # (1, 3, 32, 32)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    print(f"Input mean: {input_tensor.mean():.6f}")

    # Run just conv1 (no bias, no BN, no activation)
    with torch.no_grad():
        conv1_out = model.conv1(input_tensor)
        print(f"\nConv1 output shape: {conv1_out.shape}")  # Should be (1, 32, 32, 32)
        print(f"Conv1 output range: [{conv1_out.min():.4f}, {conv1_out.max():.4f}]")
        print(f"Conv1 output mean: {conv1_out.mean():.6f}")
        print(f"Conv1 output std: {conv1_out.std():.6f}")

        # Check specific values
        print(f"\nConv1 output[0, 0, 0, 0:5]: {conv1_out[0, 0, 0, :5]}")
        print(f"Conv1 output[0, 0, 15, 15]: {conv1_out[0, 0, 15, 15]:.6f}")

    # Now test with full forward (conv1 + bias + BN + relu + pool)
    with torch.no_grad():
        x = model.conv1(input_tensor)  # Conv
        x = x + model.conv1.bias.view(1, -1, 1, 1)  # Add bias
        x_before_bn = x.clone()
        x = model.bn1(x)  # BatchNorm
        x_before_relu = x.clone()
        x = model.leaky_relu(x)  # LeakyReLU
        x_before_pool = x.clone()
        x = model.pool1(x)  # MaxPool

        print(f"\n--- After each operation ---")
        print(
            f"After conv+bias: mean={x_before_bn.mean():.6f}, range=[{x_before_bn.min():.4f}, {x_before_bn.max():.4f}]"
        )
        print(
            f"After BN: mean={x_before_relu.mean():.6f}, range=[{x_before_relu.min():.4f}, {x_before_relu.max():.4f}]"
        )
        print(
            f"After ReLU: mean={x_before_pool.mean():.6f}, range=[{x_before_pool.min():.4f}, {x_before_pool.max():.4f}]"
        )
        print(f"After Pool: mean={x.mean():.6f}, range=[{x.min():.4f}, {x.max():.4f}]")

    print(f"\n{'=' * 70}")
    print("Next step: Export these same values from Fortran and compare!")
    print("If they match → conv1 is correct, problem is elsewhere")
    print("If they differ → conv weight loading is wrong")
    print("=" * 70)


if __name__ == "__main__":
    test_conv1_only()
