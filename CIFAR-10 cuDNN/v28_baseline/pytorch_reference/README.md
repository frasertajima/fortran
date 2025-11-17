# PyTorch Reference Implementations

This directory contains PyTorch reference implementations that **exactly match** the v28 CUDA Fortran architecture for fair performance comparisons.

## Purpose

- Validate 2× speedup claims
- Provide architectural parity reference
- Help users understand the CNN architecture
- Enable framework comparisons

## Actual results:

<img width="884" height="880" alt="Screenshot From 2025-11-17 08-51-16" src="https://github.com/user-attachments/assets/5febb43d-9385-45c5-81db-8470a0c2bbe3" />


## Architecture

All implementations use the **same CNN architecture** as v28:

```
Conv1 (in_channels→32, 3×3, padding=1)
  ↓ BatchNorm1
  ↓ ELU
  ↓ MaxPool (2×2)
Conv2 (32→64, 3×3, padding=1)
  ↓ BatchNorm2
  ↓ ELU
  ↓ MaxPool (2×2)
Conv3 (64→128, 3×3, padding=1)
  ↓ BatchNorm3
  ↓ ELU
  ↓ MaxPool (2×2)
  ↓ Flatten
FC1 (flatten_size→512)
  ↓ BatchNorm4
  ↓ ELU
  ↓ Dropout(0.3)
FC2 (512→256)
  ↓ BatchNorm5
  ↓ ELU
  ↓ Dropout(0.3)
FC3 (256→num_classes)
```

## Hyperparameters

**Optimizer**: Adam
- Learning rate: 0.001
- Betas: (0.9, 0.999)
- Weight decay: 0

**Training**:
- Batch size: 128
- Epochs: 10
- Loss: CrossEntropyLoss

**Data**:
- No augmentation (same as v28)
- Normalization: [0, 1] range

## Available Implementations

| Script | Dataset | Input Size | Classes |
|--------|---------|------------|---------|
| `cifar10_reference.py` | CIFAR-10 | 32×32×3 | 10 |
| `cifar100_reference.py` | CIFAR-100 | 32×32×3 | 100 |
| `fashion_mnist_reference.py` | Fashion-MNIST | 28×28×1 | 10 |
| `svhn_reference.py` | SVHN | 32×32×3 | 10 |

## Usage

```bash
# CIFAR-10
python cifar10_reference.py

# Fashion-MNIST
python fashion_mnist_reference.py

# CIFAR-100
python cifar100_reference.py

# SVHN
python svhn_reference.py
```

## Expected Results

| Dataset | PyTorch Time | v28 Fortran Time | Speedup | Accuracy |
|---------|--------------|------------------|---------|----------|
| CIFAR-10 | ~61s | ~31s | 2.0× | 78-79% |
| Fashion-MNIST | ~55s | ~28s | 2.0× | 91-92% |
| CIFAR-100 | ~65s | ~35s | 1.9× | 46-50% |
| SVHN | ~75s | ~40s | 1.9× | 92-93% |

**Hardware**: NVIDIA V100 GPU

## Requirements

```bash
pip install torch torchvision numpy
```

## Notes

- These implementations intentionally **do not** use:
  - Data augmentation
  - Learning rate schedules
  - Advanced optimization tricks

- This ensures **fair comparison** with v28 Fortran baseline
- Both frameworks use identical architectures and hyperparameters
