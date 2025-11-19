# Fashion-MNIST Adaptation for v28 Baseline

**Date**: 2025-11-17
**Status**: ✅ Complete and Working

## Overview

This document describes the adaptation of the v28 baseline modular training framework to support Fashion-MNIST, demonstrating the framework's flexibility in handling datasets with different input dimensions.

## Dataset Specifications

### Fashion-MNIST vs CIFAR-10

| Property | CIFAR-10 | Fashion-MNIST |
|----------|----------|---------------|
| **Image Size** | 32×32 | 28×28 |
| **Channels** | 3 (RGB) | 1 (Grayscale) |
| **Input Size** | 3072 (3×32×32) | 784 (1×28×28) |
| **Classes** | 10 | 10 |
| **Training Samples** | 50,000 | 60,000 |
| **Test Samples** | 10,000 | 10,000 |

## Key Changes Required

### 1. Configuration Parameters (`fashion_mnist_config.cuf`)

The following parameters were updated to reflect Fashion-MNIST dimensions:

```fortran
integer, parameter, public :: INPUT_CHANNELS = 1   ! Grayscale (was 3)
integer, parameter, public :: INPUT_HEIGHT = 28    ! (was 32)
integer, parameter, public :: INPUT_WIDTH = 28     ! (was 32)
integer, parameter, public :: input_size = 784     ! 1*28*28 (was 3072)
integer, parameter, public :: train_samples = 60000  ! (was 50000)
```

### 2. Critical Fix: Input Reshaping (`fashion_mnist_main.cuf:2161-2183`)

**The Problem**: The original code had hardcoded CIFAR-10 dimensions in the input reshaping kernel, which caused incorrect data layout for Fashion-MNIST.

**Before (Broken - CIFAR-10 hardcoded)**:
```fortran
associate(input_array => model%input)
    !$cuf kernel do(4)
    do i = 1, batch_size          ! N (batch)
        do j = 1, 3               ! C (channel) ❌ HARDCODED
            do k = 1, 32          ! H (height/row) ❌ HARDCODED
                do idx = 1, 32    ! W (width/column) ❌ HARDCODED
                    input_array(idx, k, j, i) = input_batch(i, (j-1)*1024 + (k-1)*32 + idx)
                    ! ❌ HARDCODED: 1024 = 32*32, and 32 = width
                end do
            end do
        end do
    end do
end associate
```

**After (Fixed - Parameterized)**:
```fortran
associate(input_array => model%input)
    !$cuf kernel do(4)
    do i = 1, batch_size              ! N (batch)
        do j = 1, INPUT_CHANNELS      ! C (channel) = 1 for Fashion-MNIST ✅
            do k = 1, INPUT_HEIGHT    ! H (height/row) = 28 ✅
                do idx = 1, INPUT_WIDTH    ! W (width/column) = 28 ✅
                    ! For Fashion-MNIST: 784 = 28*28*1
                    input_array(idx, k, j, i) = input_batch(i, (j-1)*(INPUT_HEIGHT*INPUT_WIDTH) + (k-1)*INPUT_WIDTH + idx)
                    ! ✅ PARAMETERIZED: (INPUT_HEIGHT*INPUT_WIDTH) = 784, INPUT_WIDTH = 28
                end do
            end do
        end do
    end do
end associate
```

### 3. Index Calculation Breakdown

For **CIFAR-10** (32×32×3):
- Input batch format: `(batch, 3072)` where 3072 = 3×32×32
- Index calculation: `(j-1)*1024 + (k-1)*32 + idx`
  - `(j-1)*1024`: Skip to channel j (1024 = 32×32 pixels per channel)
  - `(k-1)*32`: Skip to row k (32 = pixels per row)
  - `idx`: Column position within row

For **Fashion-MNIST** (28×28×1):
- Input batch format: `(batch, 784)` where 784 = 1×28×28
- Index calculation: `(j-1)*(INPUT_HEIGHT*INPUT_WIDTH) + (k-1)*INPUT_WIDTH + idx`
  - `(j-1)*(28*28)`: Skip to channel j (784 = 28×28 pixels per channel, but j=1 always)
  - `(k-1)*28`: Skip to row k (28 = pixels per row)
  - `idx`: Column position within row

## Files Modified/Created

### Core Files
1. **`fashion_mnist_config.cuf`** (~170 lines)
   - Dataset-specific parameters
   - Data loading from binary files
   - GPU memory allocation

2. **`fashion_mnist_main.cuf`** (~2400 lines)
   - Main training loop
   - **Critical fix**: Input reshaping kernel (lines 2161-2183)
   - Forward/backward pass
   - CNN architecture instantiation

3. **`prepare_fashion_mnist.py`** (~120 lines)
   - Downloads Fashion-MNIST dataset
   - Preprocesses and normalizes images
   - Saves to binary format for Fortran loading

4. **`compile_fashion_mnist.sh`**
   - Compilation script with proper CUDA flags
   - Links cuDNN and cuBLAS libraries

## Lessons Learned

### What Made This Adaptation Trivial

✅ **Modular design**: Common modules (`adam_optimizer.cuf`, `gpu_batch_extraction.cuf`, `random_utils.cuf`, `cuda_utils.cuf`) worked without modification

✅ **Parameterized architecture**: Using `INPUT_CHANNELS`, `INPUT_HEIGHT`, `INPUT_WIDTH` constants allowed easy dimension changes

✅ **Clear separation**: Dataset-specific code isolated in config module

### The One Critical Bug

❌ **Hardcoded dimensions in GPU kernels**: The input reshaping kernel had hardcoded loop bounds and index calculations

🔧 **Fix**: Replace hardcoded values with parameters:
- Loop bounds: `1, 3` → `1, INPUT_CHANNELS`
- Loop bounds: `1, 32` → `1, INPUT_HEIGHT` and `1, INPUT_WIDTH`
- Index calc: `(j-1)*1024 + (k-1)*32 + idx` → `(j-1)*(INPUT_HEIGHT*INPUT_WIDTH) + (k-1)*INPUT_WIDTH + idx`

### Best Practices for Future Datasets

1. **Always use parameters, never hardcode dimensions**
   - Even in GPU kernels where you think performance matters
   - Modern compilers inline these constants anyway

2. **Test with different input dimensions early**
   - CIFAR-10 (32×32×3) and Fashion-MNIST (28×28×1) caught this bug
   - Validates true modularity

3. **Document index calculations**
   - Complex array indexing should have inline comments
   - Makes bugs easier to spot during code review

## Performance Expectations

Fashion-MNIST should achieve:
- **Training accuracy**: ~90-92% (10 epochs)
- **Training speed**: ~25-30s on V100 (faster than CIFAR-10 due to smaller images)
- **Memory usage**: Lower than CIFAR-10 (784 vs 3072 input size)

## Usage

```bash
# 1. Prepare data
cd v28_baseline/datasets/fashion_mnist
python prepare_fashion_mnist.py

# 2. Compile
./compile_fashion_mnist.sh

# 3. Train
./fashion_mnist_train
```

## Integration with v28 Framework

Fashion-MNIST demonstrates the v28 baseline framework's design goals:

| Goal | Achievement |
|------|-------------|
| **Code reuse** | ✅ 100% reuse of common modules |
| **Easy adaptation** | ✅ Only config + 1 critical fix needed |
| **Performance** | ✅ Same GPU optimizations apply |
| **Modularity** | ✅ No changes to other datasets |

## Conclusion

The Fashion-MNIST adaptation validates the v28 baseline framework's modular design. Once the input reshaping bug was fixed, the entire training pipeline worked seamlessly with **zero modifications to common modules**.

**Key takeaway**: Parameterization is critical - even in GPU kernels where dimensions might seem "obvious", always use symbolic constants rather than hardcoded values.

---

**Author**: v28 Baseline Team
**Contributors**: Claude AI, Fraser Tajima
**Last Updated**: 2025-11-17
