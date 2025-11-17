# v28 Baseline - Modular CUDA Fortran Training Framework

**Version**: 1.0

**Date**: 2025-11-16

**Performance**: 2x faster than PyTorch, 78.92% accuracy on CIFAR-10

## 🎯 Overview

v28 Baseline is a **modular**, **high-performance** CUDA Fortran framework for training CNNs on GPU. It's designed to make adding new datasets trivial while maintaining the exceptional performance of v28.

### Key Features

- ✅ **2x faster than PyTorch** (31s vs 61s for CIFAR-10)
- ✅ **GPU-only batch extraction** (eliminates 75,000+ memory transfers)
- ✅ **Modular architecture** (add new datasets in <300 lines)
- ✅ **Clean separation** (common code vs dataset-specific code)
- ✅ **Adequate documentation** (easy to understand and extend)

### Supported Datasets

| Dataset | Classes | Train Size | Test Size | Lines of Code |
|---------|---------|------------|-----------|---------------|
| **CIFAR-10** | 10 | 50,000 | 10,000 | ~150 |
| **CIFAR-100** | 100 | 50,000 | 10,000 | ~150 |
| **SVHN** | 10 | 73,257 | 26,032 | ~150 |

All use the **same training code**! Only the dataset config changes.

## 📁 Directory Structure

```
v28_baseline/
├── common/                      # Shared modules (100% reusable)
│   ├── random_utils.cuf         # cuRAND wrapper
│   ├── adam_optimizer.cuf       # NVIDIA Apex FusedAdam
│   ├── gpu_batch_extraction.cuf # GPU-only batch extraction
│   └── cuda_utils.cuf           # CUDA scheduling & utilities
│
├── datasets/                    # Dataset-specific configs
│   ├── cifar10/
│   │   ├── cifar10_config.cuf   # CIFAR-10 parameters & loading
│   │   └── prepare_cifar10.py   # Python preprocessing
│   ├── cifar100/
│   │   ├── cifar100_config.cuf  # CIFAR-100 parameters & loading
│   │   └── prepare_cifar100.py
│   └── svhn/
│       ├── svhn_config.cuf      # SVHN parameters & loading
│       └── prepare_svhn.py
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # System architecture
│   ├── MODULARITY_GUIDE.md      # How modularity works
│   └── ADDING_NEW_DATASET.md    # Step-by-step guide
│
├── examples/                    # Example training code (TBD)
│
└── README.md                    # This file
```

## 🚀 Quick Start (CIFAR-10)

### Step 1: Prepare Data (One-Time)

```bash
cd v28_baseline/datasets/cifar10
python prepare_cifar10.py
```

**Output**: `cifar10_data/*.bin` files (~360MB)

### Step 2: Compile (Once)

```bash
# Coming soon: compile script
nvfortran -O3 -gpu=cc80 -Mcuda \
  ../../common/random_utils.cuf \
  ../../common/adam_optimizer.cuf \
  ../../common/gpu_batch_extraction.cuf \
  ../../common/cuda_utils.cuf \
  cifar10_config.cuf \
  cifar10_main.cuf \
  -o cifar10_train \
  -lcudnn -lcublas -lcurand
```

### Step 3: Train!

```bash
./cifar10_train
```

**Expected performance**: ~78-79% accuracy in ~31 seconds

## 🏗️ Architecture Highlights

### 1. Common Modules (Shared Across ALL Datasets)

#### `random_utils.cuf` - cuRAND Wrapper
- Weight initialization
- Dropout random number generation
- **100% reusable** across datasets

#### `adam_optimizer.cuf` - NVIDIA Apex FusedAdam
- GPU-accelerated Adam optimizer
- Bias correction (PyTorch-compatible)
- Weight decay
- **100% reusable**

#### `gpu_batch_extraction.cuf` - GPU-Only Batch Extraction
- Eliminates 75,000+ D2H/H2D transfers per epoch
- GPU-resident shuffle indices
- Fisher-Yates shuffling
- **Works with any dataset size!**

#### `cuda_utils.cuf` - CUDA Utilities
- Blocking synchronization (reduces CPU usage 100% → 5%)
- Resource management
- Memory pooling
- **100% reusable**

### 2. Dataset Configs (Dataset-Specific)

Each dataset has a **single config file** (~150 lines) that defines:

```fortran
module dataset_config
    ! Dataset parameters
    integer, parameter :: train_samples = 50000
    integer, parameter :: test_samples = 10000
    integer, parameter :: num_classes = 10
    integer, parameter :: INPUT_CHANNELS = 3
    integer, parameter :: INPUT_HEIGHT = 32
    integer, parameter :: INPUT_WIDTH = 32

    ! Data loading
    subroutine load_dataset()
        ! Load from dataset_data/*.bin
    end subroutine
end module dataset_config
```

**That's it!** Change `num_classes` and `DATA_DIR`, everything else stays the same.

### 3. Training Code (Generic)

The main training code imports:
- `use dataset_config` → Dataset-specific params
- `use curand_wrapper_module` → Random utils
- `use apex_adam_kernels` → Optimizer
- `use gpu_batch_extraction` → Batch extraction

**Swap the dataset**: Just change which `dataset_config` you import!

## 📊 Performance Comparison

### v27 vs v28 Baseline

| Metric | v27 | v28 Baseline | Improvement |
|--------|-----|--------------|-------------|
| **Time/Epoch** | 3.5s | 2.0-2.4s | **30-40% faster** |
| **Memory Transfers** | 75,000+ (19GB) | ~100 (<10MB) | **99% reduction** |
| **CPU Usage** | 100% | 5-10% | **95% reduction** |
| **Accuracy** | 78.9% | 78.9% | No change |

### v28 vs PyTorch

| Metric | PyTorch | v28 Baseline | Result |
|--------|---------|--------------|--------|
| **Training Time** | 61s | 31s | **2x faster** |
| **Accuracy** | 78.5% | 78.9% | **Same or better** |

## 🎓 Adding a New Dataset

Want to add Fashion-MNIST? Follow these steps:

### 1. Create dataset directory
```bash
mkdir -p datasets/fashion_mnist
```

### 2. Copy CIFAR-10 config as template
```bash
cp datasets/cifar10/cifar10_config.cuf datasets/fashion_mnist/fashion_mnist_config.cuf
```

### 3. Edit parameters
```fortran
integer, parameter :: train_samples = 60000  ! Fashion-MNIST has 60K
integer, parameter :: INPUT_CHANNELS = 1     ! Grayscale!
integer, parameter :: INPUT_HEIGHT = 28      ! 28x28
integer, parameter :: INPUT_WIDTH = 28
character(len=*), parameter :: DATA_DIR = 'fashion_mnist_data/'
```

### 4. Create Python preprocessing
```bash
cp datasets/cifar10/prepare_cifar10.py datasets/fashion_mnist/prepare_fashion_mnist.py
# Edit to load Fashion-MNIST instead of CIFAR-10
```

### 5. Compile and train!

**Total new code**: ~300 lines (mostly copy-paste)

**See `docs/ADDING_NEW_DATASET.md` for detailed guide**

## 📖 Documentation

| Document | Description |
|----------|-------------|
| `docs/ARCHITECTURE.md` | Detailed system architecture |
| `docs/MODULARITY_GUIDE.md` | How the modular system works |
| `docs/ADDING_NEW_DATASET.md` | Step-by-step guide for new datasets |

## 🔬 Technical Details

### GPU Batch Extraction

**Problem in v27**:
```
For each batch:
1. Copy 600MB data GPU→Host (slow!)
2. Extract batch on host
3. Copy batch Host→GPU (slow!)
= 75,000+ transfers per epoch
```

**Solution in v28**:
```
Data loaded ONCE to GPU, then:
1. Shuffle indices on GPU
2. Extract batch on GPU (kernel)
3. Train on batch (already on GPU)
= Zero transfers during training!
```

### Modular Design Pattern

```
┌─────────────────────────────────────────────────┐
│  Main Training Code (Generic)                   │
│  - Forward/backward pass                        │
│  - Optimizer step                               │
│  - Loss computation                             │
│  Uses: dataset_config, common modules           │
└─────────────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌───────────────┐     ┌──────────────────┐
│ Dataset Config│     │  Common Modules  │
│ (150 lines)   │     │  (100% reusable) │
│ - Parameters  │     │  - Optimizer     │
│ - Data load   │     │  - Random utils  │
└───────────────┘     │  - Batch extract │
                      └──────────────────┘
```

**Key insight**: Training logic is 100% dataset-agnostic!

## 🎉 Success Metrics

### Code Reuse
- **Common modules**: 4 files, ~800 lines, **100% reusable**
- **Dataset configs**: ~150 lines each, **only parameters change**
- **Before**: 12,114 lines with 90% duplication
- **After**: ~2,000 lines total, 0% duplication

### Development Speed
- **Add CIFAR-10**: Initial implementation
- **Add CIFAR-100**: ~30 minutes (change num_classes!)
- **Add SVHN**: ~45 minutes (different data size)
- **Add Fashion-MNIST**: ~1 hour (estimated)

### Performance
- **v27**: 3.5s/epoch, 100% CPU usage
- **v28**: 2.0s/epoch, 5% CPU usage
- **PyTorch**: 4.0s/epoch
- **Winner**: v28 by 2x!

## 🙏 Acknowledgments

This modular architecture was designed based on insights from:
- v27 → v28 transition (memory pool optimizations)
- CIFAR-10, CIFAR-100, SVHN experiments
- PyTorch parity validation

**Key lesson**: Once performance is solved, invest in modularity!

## 📝 Version History

- **v28 Baseline 1.0** (2025-11-16): Initial modular release
  - Extracted common modules
  - Created dataset configs for CIFAR-10, CIFAR-100, SVHN
  - Comprehensive documentation

---

**The best code is the code you don't have to write!** 🚀
