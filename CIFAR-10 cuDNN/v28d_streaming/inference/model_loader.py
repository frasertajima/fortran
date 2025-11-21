"""
Model Loader for v28 Baseline Trained Models

Loads trained CUDA Fortran models into PyTorch for inference.

Usage:
    from model_loader import load_v28_model

    # CIFAR-10 (RGB)
    model = load_v28_model('saved_models/cifar10/', in_channels=3, num_classes=10)

    # Fashion-MNIST (Grayscale)
    model = load_v28_model('saved_models/fashion_mnist/', in_channels=1, num_classes=10)

Author: v28 Baseline Team
Date: 2025-11-17
"""

import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path


class V28_CNN(nn.Module):
    """
    CNN architecture matching v28 CUDA Fortran implementation.

    This is the same architecture as the PyTorch reference, but with
    the ability to load pre-trained weights from v28 Fortran.
    """

    def __init__(self, in_channels=3, num_classes=10, input_size=32, flatten_size=None):
        super(V28_CNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size

        # Conv Block 1: in_channels → 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flatten size (override if provided)
        if flatten_size is not None:
            self.flatten_size = flatten_size
        else:
            # After 3 pooling layers: input_size → /2 → /2 → /2
            pool_size = input_size // 8
            self.flatten_size = 128 * pool_size * pool_size

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)

        # Activation - LeakyReLU to match v28 Fortran (NOT ELU)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC1
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)

        # FC2
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)

        # FC3 (output)
        x = self.fc3(x)

        return x


def load_fortran_binary(filepath, shape=None, dtype=np.float32):
    """
    Load a Fortran binary file exported by v28 model_export module.
    """
    data = np.fromfile(filepath, dtype=dtype)

    if shape is not None:
        expected_size = np.prod(shape)
        if data.size != expected_size:
            raise ValueError(f"Shape mismatch: expected {expected_size} elements, got {data.size}")

        if len(shape) == 2:
            data = data.reshape(shape, order='F')
        elif len(shape) == 4:
            data = data.reshape(shape, order='F')
        else:
            data = data.reshape(shape, order='C')

    return data


def load_v28_model(model_dir, in_channels=3, num_classes=10, input_size=32, device='cpu'):
    """
    Load a trained v28 model from exported Fortran binary files.

    Args:
        model_dir: Directory containing exported model files
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        input_size: Input image size (32 for CIFAR, 28 for Fashion-MNIST)
        device: 'cpu' or 'cuda'

    Returns:
        PyTorch model with loaded weights in eval mode
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading v28 model from: {model_dir}")

    # Auto-detect flatten_size
    fc1_weights_path = model_dir / 'fc1_weights.bin'
    if fc1_weights_path.exists():
        fc1_weights_size = fc1_weights_path.stat().st_size // 4
        flatten_size = fc1_weights_size // 512
        print(f"Auto-detected flatten_size: {flatten_size}")
    else:
        pool_size = input_size // 8
        flatten_size = 128 * pool_size * pool_size
        print(f"Using calculated flatten_size: {flatten_size}")

    # Create model
    model = V28_CNN(in_channels=in_channels, num_classes=num_classes,
                    input_size=input_size, flatten_size=flatten_size)

    # --- Internal Helper for Loading and Fixing Conv Weights ---
    def load_and_fix_conv_weights(layer_name, filename_w, filename_b, shape, target_layer):
        """
        Loads, reshapes, permutes, and flips convolution weights to match PyTorch.
        shape expectation: (H, W, In_Channels, Out_Channels)
        """
        # 1. Load raw
        w_raw = np.fromfile(model_dir / filename_w, dtype=np.float32)
        
        # 2. Reshape using F-order (Fortran default)
        # CRITICAL: Fortran exported as (H, W, C, K) or similar interleaved format
        w_reshaped = w_raw.reshape(shape, order='F')
        
        # 3. Permute axes to PyTorch format
        # From (0:H, 1:W, 2:In, 3:Out) -> (3:Out, 2:In, 1:W, 0:H)
        w_transposed = w_reshaped.transpose(3, 2, 1, 0)
        
        # 4. Spatial Flip (Fixes rotation issue)
        w_flipped = np.flip(w_transposed, axis=(2, 3)).copy()
        
        # 5. Load Bias
        b_data = load_fortran_binary(model_dir / filename_b, shape=(shape[3],))

        # 6. Assign
        target_layer.weight.data = torch.from_numpy(w_flipped)
        target_layer.bias.data = torch.from_numpy(b_data)
        print(f"Loaded {layer_name} (Dynamic Shape: {shape})")

    # --- Load Convolutional Layers ---

    print("Loading Conv Layers...")
    
    # Conv1: Input channels are dynamic (in_channels), Output=32
    load_and_fix_conv_weights(
        "Conv1", "conv1_weights.bin", "conv1_bias.bin",
        shape=(3, 3, in_channels, 32), # DYNAMIC CHANNELS HERE
        target_layer=model.conv1
    )

    # Conv2: Input=32, Output=64
    load_and_fix_conv_weights(
        "Conv2", "conv2_weights.bin", "conv2_bias.bin",
        shape=(3, 3, 32, 64),
        target_layer=model.conv2
    )

    # Conv3: Input=64, Output=128
    load_and_fix_conv_weights(
        "Conv3", "conv3_weights.bin", "conv3_bias.bin",
        shape=(3, 3, 64, 128),
        target_layer=model.conv3
    )

    # --- Load Fully Connected Layers ---
    print("Loading FC1...")
    fc1_w = load_fortran_binary(model_dir / 'fc1_weights.bin', shape=(512, model.flatten_size))
    fc1_b = load_fortran_binary(model_dir / 'fc1_bias.bin', shape=(512,))
    model.fc1.weight.data = torch.from_numpy(fc1_w)
    model.fc1.bias.data = torch.from_numpy(fc1_b)

    print("Loading FC2...")
    fc2_w = load_fortran_binary(model_dir / 'fc2_weights.bin', shape=(256, 512))
    fc2_b = load_fortran_binary(model_dir / 'fc2_bias.bin', shape=(256,))
    model.fc2.weight.data = torch.from_numpy(fc2_w)
    model.fc2.bias.data = torch.from_numpy(fc2_b)

    print("Loading FC3...")
    fc3_w = load_fortran_binary(model_dir / 'fc3_weights.bin', shape=(num_classes, 256))
    fc3_b = load_fortran_binary(model_dir / 'fc3_bias.bin', shape=(num_classes,))
    model.fc3.weight.data = torch.from_numpy(fc3_w)
    model.fc3.bias.data = torch.from_numpy(fc3_b)

    # --- Load BatchNorm Layers ---
    print("Loading BatchNorm layers...")

    def load_bn(bn_layer, prefix):
        scale = load_fortran_binary(model_dir / f'{prefix}_scale.bin')
        bias = load_fortran_binary(model_dir / f'{prefix}_bias.bin')
        running_mean = load_fortran_binary(model_dir / f'{prefix}_running_mean.bin')
        running_var = load_fortran_binary(model_dir / f'{prefix}_running_var.bin')

        bn_layer.weight.data = torch.from_numpy(scale)
        bn_layer.bias.data = torch.from_numpy(bias)
        bn_layer.running_mean.data = torch.from_numpy(running_mean)
        bn_layer.running_var.data = torch.from_numpy(running_var)

    load_bn(model.bn1, 'bn1')
    load_bn(model.bn2, 'bn2')
    load_bn(model.bn3, 'bn3')

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully!")
    print(f"   Device: {device}")
    print(f"   Architecture: {in_channels}ch → Conv32 → Conv64 → Conv128 → FC512 → FC256 → FC{num_classes}")

    return model


def load_metadata(model_dir):
    """
    Load model metadata from the export directory.
    """
    model_dir = Path(model_dir)
    metadata_file = model_dir / 'model_metadata.txt'

    if not metadata_file.exists():
        return None

    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            if 'Dataset:' in line:
                metadata['dataset'] = line.split(':', 1)[1].strip()
            elif 'Test Accuracy:' in line:
                acc_str = line.split(':', 1)[1].strip()
                metadata['accuracy'] = float(acc_str.replace('%', ''))
            elif 'Epochs Trained:' in line:
                metadata['epochs'] = int(line.split(':', 1)[1].strip())
            elif 'Export Date:' in line:
                metadata['export_date'] = line.split(':', 1)[1].strip()

    return metadata


if __name__ == '__main__':
    print("v28 Model Loader")
    print("=" * 70)
    print("Example Usage:")
    print("  model = load_v28_model('saved_models/cifar10/', in_channels=3)")
    print("  model = load_v28_model('saved_models/fashion_mnist/', in_channels=1)")