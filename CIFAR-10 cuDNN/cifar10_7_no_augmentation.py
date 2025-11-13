#!/usr/bin/env python3
"""
CIFAR-10 Baseline Test - NO AUGMENTATION
========================================
Modified from cifar10-7.ipynb to remove all data augmentation
This will establish the Python baseline accuracy without augmentation
for direct comparison with Fortran implementation.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import version 7 TensorCore
try:
    from tensor_core_wrapper7 import TensorCore, TensorCoreLinear, get_global_tensorcore

    print("✅ Imported tensor_core_wrapper7 with TensorCoreLinear")
    TENSORCORE_AVAILABLE = True
except ImportError:
    print("❌ Could not import tensor_core_wrapper7!")
    print("   Please ensure tensor_core_wrapper7 is available")
    sys.exit(1)

# CIFAR-10 classes
CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def set_debug_mode(enabled):
    """Set debug mode for TensorCore operations"""
    try:
        tc = TensorCore(debug_mode=enabled)
        print(f"🔧 TensorCore debug mode: {'enabled' if enabled else 'disabled'}")
    except Exception as e:
        print(f"⚠️ Could not set debug mode: {e}")


def load_cifar10_data_no_augmentation(batch_size=128, num_workers=4):
    """
    Load CIFAR-10 with NO AUGMENTATION - only normalization
    """
    print("📁 Loading CIFAR-10 dataset with NO AUGMENTATION...")

    # NO AUGMENTATION - only normalization for both train and test
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Same transform for test (no augmentation needed anyway)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"✅ CIFAR-10 loaded with NO AUGMENTATION:")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    print(f"   Batch size: {batch_size}")
    print(
        f"   Only normalization applied: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)"
    )

    return train_loader, test_loader


# =============================================================================
# MODEL ARCHITECTURE - Same as Fortran version
# =============================================================================


class CIFAR10NetTensorCoreBaseline(nn.Module):
    """
    CNN model matching the Fortran architecture exactly
    No augmentation dependency - baseline comparison
    """

    def __init__(self):
        super(CIFAR10NetTensorCoreBaseline, self).__init__()

        # Convolutional layers (matching Fortran)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate feature size after conv layers: 32->16->8->4, 4*4*128 = 2048
        self.feature_size = 4 * 4 * 128

        # Fully connected layers using TensorCore
        self.fc1 = TensorCoreLinear(self.feature_size, 512)
        self.fc2 = TensorCoreLinear(512, 256)
        self.fc3 = TensorCoreLinear(256, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1: Conv -> BN -> LeakyReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)  # 32x32 -> 16x16

        # Conv block 2: Conv -> BN -> LeakyReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)  # 16x16 -> 8x8

        # Conv block 3: Conv -> BN -> LeakyReLU -> Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)  # 8x8 -> 4x4

        # Flatten and FC layers
        x = torch.flatten(x, 1)  # Shape: (batch_size, 2048)

        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Progress update every 100 batches
        if batch_idx % 100 == 0:
            print(
                f"   Batch {batch_idx:3d}/{len(train_loader)} "
                f"Loss: {loss.item():.6f} "
                f"Acc: {100.0 * correct / total:.3f}%"
            )

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, epoch_time


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Per-class accuracy
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total

    # Calculate per-class accuracies
    class_accuracies = []
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            class_accuracies.append(class_acc)
            print(f"   {CIFAR10_CLASSES[i]:>6}: {class_acc:6.2f}%")
        else:
            class_accuracies.append(0.0)

    return test_loss, accuracy, 0, class_accuracies  # 0 for eval_time compatibility


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def main_baseline_training(num_epochs=15, batch_size=128, lr=0.001):
    """
    Main training function - NO AUGMENTATION BASELINE
    """
    print("🚀 CIFAR-10 BASELINE TRAINING - NO AUGMENTATION")
    print("=" * 60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Disable TensorCore debug for speed
    set_debug_mode(False)

    # Load data WITHOUT augmentation
    train_loader, test_loader = load_cifar10_data_no_augmentation(batch_size=batch_size)

    # Initialize model (same architecture as successful version 6)
    torch.manual_seed(42)  # Reproducibility
    model = CIFAR10NetTensorCoreBaseline().to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Information:")
    print(f"   Parameters: {params:,}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Architecture: Same as Fortran version")
    print(f"   Augmentation: NONE (baseline test)")

    # Optimizer and scheduler (matching Python version but no augmentation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_times": [],
    }

    best_test_acc = 0.0
    total_time = 0.0

    print(f"\n🏋️ Starting Baseline Training (NO AUGMENTATION)...")

    for epoch in range(1, num_epochs + 1):
        print(f"\n📈 EPOCH {epoch}/{num_epochs}")
        print("-" * 60)

        # Training
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Evaluation
        test_loss, test_acc, eval_time, class_accs = evaluate_model(
            model, test_loader, criterion, device
        )

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"📉 Learning rate: {current_lr:.6f}")

        # 🔍 DIAGNOSTIC: Print BN1 running variance to compare with Fortran
        if epoch == 1:
            bn1_running_var = model.bn1.running_var.cpu().numpy()
            print(f"\n🔬 DETAILED BatchNorm Diagnostics (Epoch 1):")
            print(f"   BN1 running_var (all 32 channels):")
            for i in range(0, 32, 8):
                print(f"   {bn1_running_var[i : i + 8]}")
            print(
                f"   BN1 running_var[13,21,30] (Fortran outliers): {bn1_running_var[12]:.4f} {bn1_running_var[20]:.4f} {bn1_running_var[29]:.4f}"
            )
            print(f"   Expected: var ≈ 0.9-1.1 for most channels\n")

        # Track best performance
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"💾 New best accuracy: {best_test_acc:.2f}%")
            # Save the best model
            torch.save(model.state_dict(), "model_cifar10_baseline.pth")
            print(f"   ✅ Saved model to model_cifar10_baseline.pth")

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(epoch_time)

        total_time += epoch_time

        print("-" * 60)

    # Final results
    print(f"\n🎉 BASELINE Training Completed (NO AUGMENTATION)!")
    print("=" * 60)
    print(f"🏆 Best test accuracy: {best_test_acc:.2f}%")
    print(f"⏱️ Total training time: {total_time:.1f}s")
    print(f"📊 Average time per epoch: {total_time / num_epochs:.1f}s")

    # Compare with expected results
    print(f"\n📊 Baseline Performance Analysis:")
    print(f"   Python baseline (no aug): {best_test_acc:.2f}%")
    print(f"   Fortran current (no aug): ~58.7%")
    print(f"   Gap to close: {abs(best_test_acc - 58.7):.1f} percentage points")

    if best_test_acc > 65.0:
        print("✅ GOOD: Baseline shows Python can achieve higher accuracy")
        print("   The gap is likely in training implementation, not augmentation")
    elif best_test_acc < 60.0:
        print("⚠️ CONCERNING: Python baseline is also low")
        print("   This suggests model architecture or hyperparameter issues")
    else:
        print("📊 COMPARABLE: Python and Fortran baselines are similar")
        print("   Focus should be on augmentation implementation")

    return history, best_test_acc


if __name__ == "__main__":
    # Run baseline test
    print("🧪 Running CIFAR-10 Baseline Test (NO AUGMENTATION)")
    print("This will establish the Python baseline for comparison")
    print()

    history, best_acc = main_baseline_training(num_epochs=15, batch_size=128, lr=0.001)

    print(f"\n📋 Summary:")
    print(f"   Python baseline (no augmentation): {best_acc:.2f}%")
    print(f"   Any remaining gap indicates fundamental training differences")
# works with py313 but crashes with py314!
