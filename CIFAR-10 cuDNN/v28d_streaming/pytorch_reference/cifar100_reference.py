#!/usr/bin/env python3
"""
CIFAR-100 PyTorch Reference Implementation

This implementation exactly matches the v28 CUDA Fortran architecture
for fair performance comparison.

Expected Performance (V100):
- Time: ~65 seconds (10 epochs)
- Accuracy: 46-50%

Comparison to v28 Fortran:
- v28 Fortran: ~35 seconds (1.9× faster)
- Same architecture, same hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np


class V28_CNN(nn.Module):
    """
    CNN architecture matching v28 CUDA Fortran implementation.
    Adapted for CIFAR-100 (100 classes).

    Architecture:
    - 3 conv blocks (Conv → BN → ELU → MaxPool)
    - 3 fully connected layers with BatchNorm, ELU, Dropout
    """

    def __init__(self, in_channels=3, num_classes=100):
        super(V28_CNN, self).__init__()

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

        # Calculate flatten size for 32×32 input
        # After 3 pooling layers: 32 → 16 → 8 → 4
        self.flatten_size = 128 * 4 * 4  # 2048

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn4 = nn.BatchNorm1d(512, eps=1e-5, momentum=0.1)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256, eps=1e-5, momentum=0.1)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, num_classes)

        # Activation
        self.elu = nn.ELU()

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC1
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.dropout1(x)

        # FC2
        x = self.fc2(x)
        x = self.bn5(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # FC3 (output)
        x = self.fc3(x)

        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    print("=" * 70)
    print("CIFAR-100 PyTorch Reference Implementation (v28 Architecture)")
    print("=" * 70)

    # Hyperparameters (matching v28 Fortran)
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data loading (no augmentation, matching v28)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
    ])

    print("\nLoading CIFAR-100 dataset...")
    train_dataset = datasets.CIFAR100(root='./data', train=True,
                                       download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False,
                                      download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model
    model = V28_CNN(in_channels=3, num_classes=100).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer (matching v28)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                          betas=(0.9, 0.999), weight_decay=0.0)

    # Training
    print("\n" + "=" * 70)
    print("Starting Training (10 epochs, batch size 128)")
    print("=" * 70)

    start_time = time.time()
    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"\nEpoch {epoch:2d}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:6.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}  Test Acc:  {test_acc:6.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

    total_time = time.time() - start_time

    # Final results
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)
    print(f"🏆 Best Test Accuracy:  {best_test_acc:.2f}%")
    print(f"📊 Final Train Accuracy: {train_acc:.2f}%")
    print(f"📊 Final Test Accuracy:  {test_acc:.2f}%")
    print(f"⏱️  Total Time:    {total_time:.1f}s")
    print(f"⏱️  Avg Time/Epoch: {total_time/num_epochs:.1f}s")
    print("=" * 70)

    # Comparison
    print("\n📈 Comparison to v28 CUDA Fortran:")
    print(f"   PyTorch:       ~{total_time:.0f}s (this run)")
    print(f"   v28 Fortran:   ~35s (reported)")
    print(f"   Speedup:       {total_time/35:.1f}× faster (Fortran)")
    print("=" * 70)


if __name__ == '__main__':
    main()
