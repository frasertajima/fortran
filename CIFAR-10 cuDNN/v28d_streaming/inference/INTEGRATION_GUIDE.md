# Model Export Integration Guide

**Step-by-step instructions for adding model export to your v28 training scripts**

---

## 📋 Overview

This guide shows you how to add model export functionality to any v28 dataset training script. We'll use Fashion-MNIST as an example, but the steps are identical for all datasets.

**Quick Start**: Dataset-specific inference notebooks are ready to use:
- `datasets/fashion_mnist/fashion_mnist_inference.ipynb`
- `datasets/cifar10/cifar10_inference.ipynb`
- `datasets/cifar100/cifar100_inference.ipynb`
- `datasets/svhn/svhn_inference.ipynb`

---

## ✅ Step 1: Add Model Export Module to Compilation

**File**: `compile_fashion_mnist.sh` (or your dataset's compile script)

**Find this section**:
```bash
nvfortran -O3 -gpu=cc80 -Mcuda \
  ../../common/random_utils.cuf \
  ../../common/adam_optimizer.cuf \
  ../../common/gpu_batch_extraction.cuf \
  ../../common/cuda_utils.cuf \
  fashion_mnist_config.cuf \
  fashion_mnist_main.cuf \
  -o fashion_mnist_train \
  -lcudnn -lcublas
```

**Add one line**:
```bash
nvfortran -O3 -gpu=cc80 -Mcuda \
  ../../common/random_utils.cuf \
  ../../common/adam_optimizer.cuf \
  ../../common/gpu_batch_extraction.cuf \
  ../../common/cuda_utils.cuf \
  ../../common/model_export.cuf \    # ← ADD THIS LINE
  fashion_mnist_config.cuf \
  fashion_mnist_main.cuf \
  -o fashion_mnist_train \
  -lcudnn -lcublas
```

**That's it for compilation!**

---

## ✅ Step 2: Import Model Export Module

**File**: `fashion_mnist_main.cuf` (your main training file)

**Find the module imports** (usually at the top):
```fortran
program train_fashion_mnist
    use cudafor
    use iso_c_binding
    use dataset_config
    use random_utils
    use adam_optimizer
    use gpu_batch_extraction
    use cuda_utils
```

**Add one line**:
```fortran
program train_fashion_mnist
    use cudafor
    use iso_c_binding
    use dataset_config
    use random_utils
    use adam_optimizer
    use gpu_batch_extraction
    use cuda_utils
    use model_export          ! ← ADD THIS LINE
```

---

## ✅ Step 3: Export Model After Training

**File**: `fashion_mnist_main.cuf`

**Find the end of your training loop** (after final epoch, before `end program`):

```fortran
    ! Training loop complete
    print *, ""
    print *, "======================================================================"
    print *, "🎉 TRAINING COMPLETED!"
    print *, "======================================================================"
    print *, "🏆 Best Test Accuracy: ", best_test_acc, "%"
    print *, "======================================================================"

    ! Cleanup
    deallocate(...)

end program train_fashion_mnist
```

**Add export code before cleanup**:

```fortran
    ! Training loop complete
    print *, ""
    print *, "======================================================================"
    print *, "🎉 TRAINING COMPLETED!"
    print *, "======================================================================"
    print *, "🏆 Best Test Accuracy: ", best_test_acc, "%"
    print *, "======================================================================"

    ! ====================================================================
    ! EXPORT MODEL FOR INFERENCE
    ! ====================================================================
    call create_export_directory("saved_models/fashion_mnist/", &
                                 "Fashion-MNIST", &
                                 best_test_acc, &
                                 num_epochs, &
                                 "2025-11-17")

    call export_model_generic("saved_models/fashion_mnist/", &
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
    deallocate(...)

end program train_fashion_mnist
```

---

## 📝 Template for Other Datasets

**For CIFAR-10/100/SVHN**, just change the dataset name and path:

```fortran
! For CIFAR-10:
call create_export_directory("saved_models/cifar10/", &
                             "CIFAR-10", &
                             best_test_acc, num_epochs, "2025-11-17")
call export_model_generic("saved_models/cifar10/", &
    ! ... same parameters ...

! For CIFAR-100:
call create_export_directory("saved_models/cifar100/", &
                             "CIFAR-100", &
                             best_test_acc, num_epochs, "2025-11-17")
call export_model_generic("saved_models/cifar100/", &
    ! ... same parameters ...

! For SVHN:
call create_export_directory("saved_models/svhn/", &
                             "SVHN", &
                             best_test_acc, num_epochs, "2025-11-17")
call export_model_generic("saved_models/svhn/", &
    ! ... same parameters ...
```

**Everything else is identical!**

---

## 🔍 What Gets Created

After training completes, you'll see:

```
saved_models/
└── fashion_mnist/
    ├── model_metadata.txt        # Human-readable info
    ├── conv1_weights.bin         # 19 binary files total
    ├── conv1_bias.bin
    ├── conv2_weights.bin
    ├── conv2_bias.bin
    └── ... (all weights and BatchNorm params)
```

**Console output**:
```
======================================================================
Exporting Model Weights to: saved_models/fashion_mnist/
======================================================================
Exporting Conv1...
Exporting Conv2...
Exporting Conv3...
Exporting FC1...
Exporting FC2...
Exporting FC3...
Exporting BatchNorm layers...

✅ Model export complete!
======================================================================
```

---

## 🎯 Summary Checklist

For each dataset, you need to:

- [ ] **Step 1**: Add `../../common/model_export.cuf \` to compile script
- [ ] **Step 2**: Add `use model_export` to module imports
- [ ] **Step 3**: Add `create_export_directory()` call before cleanup
- [ ] **Step 4**: Add `export_model_generic()` call with model parameters
- [ ] **Step 5**: Recompile and run training

**That's it!** Three simple additions to your existing code.

---

## 🚀 Usage After Export

Once exported, load the model in Python:

```python
from inference.model_loader import load_v28_model

# Fashion-MNIST (28x28x1, 10 classes)
model = load_v28_model('saved_models/fashion_mnist/',
                       in_channels=1, num_classes=10, input_size=28)

# CIFAR-10 (32x32x3, 10 classes)
model = load_v28_model('saved_models/cifar10/',
                       in_channels=3, num_classes=10, input_size=32)

# CIFAR-100 (32x32x3, 100 classes)
model = load_v28_model('saved_models/cifar100/',
                       in_channels=3, num_classes=100, input_size=32)

# SVHN (32x32x3, 10 classes)
model = load_v28_model('saved_models/svhn/',
                       in_channels=3, num_classes=10, input_size=32)
```

---

## 💡 Tips

1. **Export happens automatically** after training - no need to run a separate command
2. **Directory is created** automatically if it doesn't exist
3. **Old exports are overwritten** - rename the directory if you want to keep multiple versions
4. **Metadata file** contains training info (accuracy, epochs, date)
5. **All files are needed** - don't delete any of the 19 binary files (6 conv + 6 FC + 12 BatchNorm - only 3 BN layers)

---

## 🐛 Troubleshooting

**Compilation error: "module model_export not found"**
- Check that `../../common/model_export.cuf` path is correct relative to your dataset directory
- Verify the file exists at `v28_baseline/common/model_export.cuf`

**Runtime error: "cannot open saved_models/..."**
- The directory will be created automatically (uses `mkdir -p`)
- Make sure you have write permissions in the working directory

**Missing model parameters**
- If your model structure is different, you may need to adjust the parameter list
- All parameters should match your model's actual structure

---

**Author**: v28 Baseline Team
**Date**: 2025-11-17
**Status**: Production Ready
