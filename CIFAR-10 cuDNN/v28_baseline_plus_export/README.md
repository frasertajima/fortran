# CIFAR-10 cuDNN - Modular CUDA fortran training with export to python

**High-performance, modular CNN training framework in CUDA Fortran**

## 🎯 What's Here

This repository contains the **v28 Baseline plus export modules** - a modular CUDA Fortran framework for training CNNs on GPU.

### Key Features

- ✅ **4× faster than PyTorch** (31s vs 146s on CIFAR-10)
- ✅ **Fully modular** - add new datasets in <2 hours
- ✅ **Validated on 4 datasets** - CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST
- ✅ **Comprehensive documentation** - design, architecture, and adaptation guides


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

1. **`v28_baseline_plus_export/MODULAR_ADAPTATION_GUIDE.md`** - High-level overview
2. **`v28_baseline_plus_export/README.md`** - Quick start guide
3. **`v28_baseline_plus_export/docs/ARCHITECTURE.md`** - System design
4. **`v28_baseline_plus_export/FASHION_MNIST_ADAPTATION.md`** - Real-world case study

## 🏆 Why v28 Baseline?

**Before**: 12,114 lines of duplicated code across 3 datasets (90% duplication)
**After**: 1,500 lines total (0% duplication, 100% reusable)

**Performance**: Same 4× speedup over PyTorch
**Modularity**: PyTorch-level modularity achieved!

## Use Jupyter notebook as normal for intereference and other statistical tests:

<img width="990" height="459" alt="Screenshot From 2025-11-19 15-35-47" src="https://github.com/user-attachments/assets/59a896f8-07e6-4e8c-a0bc-de69fcfbbd83" />
<img width="981" height="844" alt="Screenshot From 2025-11-19 14-51-03" src="https://github.com/user-attachments/assets/c684c489-78ca-4f18-b0d2-8e461e8ec9d9" />
<img width="1325" height="973" alt="Screenshot From 2025-11-19 13-19-33" src="https://github.com/user-attachments/assets/ccdf7f37-28e9-464e-b890-5dcba02b834e" />
<img width="1325" height="1031" alt="Screenshot From 2025-11-19 12-52-09" src="https://github.com/user-attachments/assets/f07d8eb2-9e5b-4746-8b88-01ae81ce2111" />


## 🎓 Learn More

See `v28_baseline_plus_export/` for complete documentation including:
- How the modularity works
- How to adapt to new datasets
- Design principles and best practices
- Performance characteristics

---

**Repository**: https://github.com/frasertajima/CIFAR-10
**Status**: ✅ Production-ready
**Last Updated**: 2025-11-19
