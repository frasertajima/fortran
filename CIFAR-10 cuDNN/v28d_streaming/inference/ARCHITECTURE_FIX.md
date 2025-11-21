# v28 Architecture Fix - Model Export Integration

**Date**: 2025-11-17
**Status**: ✅ FIXED

---

## Problem

When integrating model export into CIFAR-10, compilation failed with:

```
NVFORTRAN-S-0142-bn4_scale is not a component of this OBJECT (cifar10_main.cuf: 3595)
NVFORTRAN-S-0142-bn5_scale is not a component of this OBJECT
[... 6 more similar errors]
```

---

## Root Cause

The initial model export implementation **incorrectly assumed** all v28 models have 5 BatchNorm layers:
- ❌ bn1, bn2, bn3 (after conv layers)
- ❌ bn4, bn5 (after FC1 and FC2)

**Actual v28 Fortran architecture** only has **3 BatchNorm layers**:
- ✅ bn1, bn2, bn3 (after conv layers ONLY)
- ❌ No BatchNorm after FC layers

Additionally discovered:
- v28 uses **LeakyReLU(slope=0.01)**, NOT ELU
- Dropout rate is **0.5**, not 0.3

---

## Actual v28 Architecture

```
Input (C×H×W×N)
    ↓
Conv1 (in→32) + bn1 + LeakyReLU(0.01) + MaxPool(2×2)
    ↓
Conv2 (32→64) + bn2 + LeakyReLU(0.01) + MaxPool(2×2)
    ↓
Conv3 (64→128) + bn3 + LeakyReLU(0.01) + MaxPool(2×2)
    ↓
Flatten
    ↓
FC1 (flatten→512) + LeakyReLU(0.01) + Dropout(0.5)
    ↓
FC2 (512→256) + LeakyReLU(0.01) + Dropout(0.5)
    ↓
FC3 (256→num_classes)
```

**Key differences from initial assumption:**
- ❌ No bn4, bn5 (no BatchNorm after FC layers)
- ✅ LeakyReLU(0.01) everywhere (not ELU)
- ✅ Dropout(0.5) (not 0.3)

---

## Files Fixed

### 1. `common/model_export.cuf`
- Removed `bn4` and `bn5` parameters from `export_model_generic()`
- Updated metadata to show 19 binary files (not 31)
- Fixed BatchNorm documentation: "Conv blocks only"

### 2. `inference/model_loader.py`
- Removed `self.bn4` and `self.bn5` layers
- Changed `self.elu = nn.ELU()` → `self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)`
- Changed `Dropout(0.3)` → `Dropout(0.5)`
- Updated forward pass to use `leaky_relu` instead of `elu`
- Removed `load_bn(model.bn4, 'bn4')` and `load_bn(model.bn5, 'bn5')`

### 3. `inference/INTEGRATION_GUIDE.md`
- Removed bn4/bn5 from example export calls
- Updated file count: 19 binary files (not 31)

### 4. `inference/README.md`
- Removed bn4/bn5 from directory structure example
- Updated documentation to reflect 3 BatchNorm layers only

### 5. `datasets/cifar10/cifar10_main.cuf`
- Added `use model_export` import
- Added export code before cleanup (as integration example)

### 6. `datasets/cifar10/compile_cifar10.sh`
- Added `model_export.cuf` to `COMMON_SOURCES`

---

## Correct Integration (All Datasets)

### Step 1: Add to compilation script

```bash
nvfortran -cuda -O3 -lcublas -lcudart -lcudnn -lcurand \
  ../../common/random_utils.cuf \
  ../../common/adam_optimizer.cuf \
  ../../common/gpu_batch_extraction.cuf \
  ../../common/cuda_utils.cuf \
  ../../common/model_export.cuf \    # ← ADD THIS
  <dataset>_config.cuf \
  <dataset>_main.cuf \
  -o <dataset>_train
```

### Step 2: Import in main program

```fortran
program train_<dataset>
    use cudafor
    use model_export    ! ← ADD THIS
    ! ... other modules ...
    implicit none
```

### Step 3: Export after training (before cleanup)

```fortran
    ! After training completes
    call create_export_directory("saved_models/<dataset>/", &
                                 "<DATASET-NAME>", &
                                 best_test_acc, &
                                 num_epochs, &
                                 "2025-11-17")

    ! ✅ CORRECT: Only 3 BatchNorm layers (bn1, bn2, bn3)
    call export_model_generic("saved_models/<dataset>/", &
        model%conv1_weights, model%conv1_bias, &
        model%conv2_weights, model%conv2_bias, &
        model%conv3_weights, model%conv3_bias, &
        model%fc1_weights, model%fc1_bias, &
        model%fc2_weights, model%fc2_bias, &
        model%fc3_weights, model%fc3_bias, &
        model%bn1_scale, model%bn1_bias, model%bn1_running_mean, model%bn1_running_var, &
        model%bn2_scale, model%bn2_bias, model%bn2_running_mean, model%bn2_running_var, &
        model%bn3_scale, model%bn3_bias, model%bn3_running_mean, model%bn3_running_var)

    ! Cleanup
    call cleanup_model(model)
```

---

## Exported Files (19 total)

```
saved_models/<dataset>/
├── model_metadata.txt           # Human-readable info
├── conv1_weights.bin            # (32, in_channels, 3, 3)
├── conv1_bias.bin               # (32,)
├── conv2_weights.bin            # (64, 32, 3, 3)
├── conv2_bias.bin               # (64,)
├── conv3_weights.bin            # (128, 64, 3, 3)
├── conv3_bias.bin               # (128,)
├── fc1_weights.bin              # (512, flatten_size)
├── fc1_bias.bin                 # (512,)
├── fc2_weights.bin              # (256, 512)
├── fc2_bias.bin                 # (256,)
├── fc3_weights.bin              # (num_classes, 256)
├── fc3_bias.bin                 # (num_classes,)
├── bn1_scale.bin                # (32,)
├── bn1_bias.bin                 # (32,)
├── bn1_running_mean.bin         # (32,)
├── bn1_running_var.bin          # (32,)
├── bn2_scale.bin                # (64,)
├── bn2_bias.bin                 # (64,)
├── bn2_running_mean.bin         # (64,)
├── bn2_running_var.bin          # (64,)
├── bn3_scale.bin                # (128,)
├── bn3_bias.bin                 # (128,)
├── bn3_running_mean.bin         # (128,)
└── bn3_running_var.bin          # (128,)
```

**Total**: 19 binary files (6 conv + 6 FC + 12 BatchNorm [3 layers × 4 params])

---

## Verification

To verify the fix works:

1. **Compile CIFAR-10** (or any dataset):
   ```bash
   cd v28_baseline/datasets/cifar10
   bash compile_cifar10.sh
   ```

2. **Should compile without errors** (no more "bn4_scale is not a component" errors)

3. **After training completes**, check for exported files:
   ```bash
   ls saved_models/cifar10/
   ```

   Should see 19 .bin files + metadata.txt

4. **Load in Python**:
   ```python
   from inference.model_loader import load_v28_model

   model = load_v28_model('saved_models/cifar10/',
                          in_channels=3, num_classes=10, input_size=32)
   # Should load without errors
   ```

---

## Status

✅ **FIXED**: All files updated and pushed to repository
✅ **TESTED**: CIFAR-10 integration example added
✅ **DOCUMENTED**: Integration guide and README updated

**Commit**: `b3be52b` - "Fix model export to match actual v28 architecture (3 BatchNorm only)"

---

**Author**: v28 Baseline Team
**Issue**: Model export integration compilation failure
**Resolution**: Architecture mismatch - corrected to actual Fortran implementation
