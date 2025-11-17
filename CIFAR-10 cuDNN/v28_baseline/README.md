# CIFAR-10 cuDNN - Modular CUDA Fortran Training

**High-performance, modular CNN training framework in CUDA Fortran**



https://github.com/user-attachments/assets/2d4ee22e-4a07-47cd-9afe-81e2bfb617fb



## 🎯 What's Here

This repository contains the **v28 Baseline** - a production-ready, modular CUDA Fortran framework for training CNNs on GPU.

### Key Features

- ✅ **2× faster than PyTorch** (31s vs 61s on CIFAR-10)
- ✅ **Fully modular** - add new datasets in <2 hours
- ✅ **Validated on 4 datasets** - CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST
- ✅ **Comprehensive documentation** - design, architecture, and adaptation guides

## 📁 Repository Structure

```
v28_baseline/              # Main framework (START HERE!)
├── README.md              # Quick start guide
├── MODULAR_ADAPTATION_GUIDE.md  # How to adapt to new datasets
├── V28_BASELINE_SUMMARY.md      # Project summary
├── FASHION_MNIST_ADAPTATION.md  # Case study
│
├── common/                # Reusable modules (885 lines, 100% reusable)
│   ├── random_utils.cuf
│   ├── adam_optimizer.cuf
│   ├── gpu_batch_extraction.cuf
│   └── cuda_utils.cuf
│
├── datasets/              # Dataset configs (~150 lines each)
│   ├── cifar10/
│   ├── cifar100/
│   ├── svhn/
│   ├── fashion_mnist/
│   └── oxford_flowers/
│
└── docs/                  # Technical documentation
    ├── ARCHITECTURE.md
    ├── MODULARITY_GUIDE.md
    └── ADDING_NEW_DATASET.md
```

## 🚀 Quick Start

```bash
# 1. Navigate to framework
cd v28_baseline

# 2. Read the overview
cat MODULAR_ADAPTATION_GUIDE.md

# 3. Try CIFAR-10
cd datasets/cifar10
python prepare_cifar10.py
./compile_cifar10.sh
./cifar10_train
```

## 📊 Validated Results

| Dataset | Accuracy | Time (V100) | Lines of Code |
|---------|----------|-------------|---------------|
| CIFAR-10 | 78.92% | 31s | ~150 |
| CIFAR-100 | 46-50% | ~35s | ~150 |
| SVHN | 92-93% | ~40s | ~150 |
| Fashion-MNIST | 92.09% | 28s | ~150 |

## 📖 Documentation

Start with these documents in order:

1. **`v28_baseline/MODULAR_ADAPTATION_GUIDE.md`** - High-level overview
2. **`v28_baseline/README.md`** - Quick start guide
3. **`v28_baseline/docs/ARCHITECTURE.md`** - System design
4. **`v28_baseline/FASHION_MNIST_ADAPTATION.md`** - Real-world case study

## 🏆 Why v28 Baseline?

**Before**: 12,114 lines of duplicated code across 3 datasets (90% duplication)
**After**: 1,500 lines total (0% duplication, 100% reusable)

**Performance**: Same 2× speedup over PyTorch
**Modularity**: PyTorch-level modularity achieved!

## 🎓 Learn More

See `v28_baseline/` for complete documentation including:
- How the modularity works
- How to adapt to new datasets
- Design principles and best practices
- Performance characteristics

---

**Repository**: https://github.com/frasertajima/CIFAR-10
**Status**: ✅ Production-ready
**Last Updated**: 2025-11-17
