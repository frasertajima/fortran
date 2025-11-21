# Fashion-MNIST Flatten Size Bug - Post-Mortem

**Date**: 2025-11-17
**Status**: 🔴 **CRITICAL BUG FIXED - RETRAIN REQUIRED**

---

## The Bug

Fashion-MNIST training code had **8 hardcoded instances** of `CONV3_FILTERS * 4 * 4 = 2048`, which is correct for 32×32 images (CIFAR-10) but **wrong for 28×28 images** (Fashion-MNIST).

### Expected vs Actual

**Expected flatten_size for 28×28**:
```
28 → 14 (pool) → 7 (pool) → 3 (pool)
flatten_size = 128 × 3 × 3 = 1,152
```

**Hardcoded in training**:
```
flatten_size = 128 × 4 × 4 = 2,048
```

**Discrepancy**: 896 extra values (2048 - 1152)

---

## What Happened During Training

The model was trained with a **catastrophic memory bug**:

1. **FC1 layer allocated** for 2,048 inputs
2. **Only 1,152 real features** copied from pool3_out
3. **Remaining 896 values** = **uninitialized memory garbage**
4. Model **overfitted to random noise** instead of image features
5. Achieved 92.09% "training accuracy" (meaningless)
6. Achieved **10.19% test accuracy** (worse than random guessing!)

### Memory Layout During Buggy Training

```
FC1 Input Buffer (2048 values):
├─ [0-1151]    Real image features from pool3  ✅
└─ [1152-2047] Random garbage from memory      💥 BUG!
```

The model learned to classify based on:
- **56% real features** (1152/2048)
- **44% random noise** (896/2048)

This is why the model predicts "Shirt" for 99% of inputs - it learned nothing useful.

---

## Why Test Inference Failed

During inference with PyTorch:
1. Real data produces **1×1152** features (correct)
2. Model expects **1×2048** features (from buggy training)
3. We **pad with zeros** to match: `[1152 real features | 896 zeros]`
4. Model expects **random noise** in positions 1153-2048
5. Gets **zeros** instead → all learned patterns broken
6. Defaults to predicting "Shirt" (99.1% of the time)

### Test Results (Unusable Model)

```
Test Accuracy: 10.19% (1019/10000)

Per-Class Accuracy:
  Shirt:       99.10% (only class it predicts)
  Bag:          2.50%
  T-shirt:      0.30%
  All others:   0.00%

Most Confused:
  Trouser  → Shirt: 100.0%
  Ankle boot → Shirt: 99.9%
  Dress → Shirt: 99.9%
```

The 92.09% claimed training accuracy was **completely bogus**.

---

## The Fix

Replaced all 8 hardcoded instances with parameterized calculations:

```fortran
! Before (WRONG for 28×28):
allocate(model%flatten(batch_size, CONV3_FILTERS * 4 * 4))

! After (CORRECT for any input size):
allocate(model%flatten(batch_size, CONV3_FILTERS * ((INPUT_HEIGHT/4)/2) * ((INPUT_WIDTH/4)/2)))
```

### Fixed Locations

1. **Line 1354**: FC1 diagnostics - total_weights calculation
2. **Line 1357**: FC1 diagnostics - host array allocation
3. **Line 1422**: grad_fc1_weights allocation
4. **Line 1443**: flatten buffer allocation (main bug)
5. **Line 1456**: grad_fc1_weights allocation (duplicate)
6. **Line 1495**: m_fc1_weights allocation (Adam first moment)
7. **Line 1503**: v_fc1_weights allocation (Adam second moment)
8. **Line 2720**: cublasSgemm call for FC1 gradient computation

---

## How to Retrain

The existing Fashion-MNIST model is **completely unusable** and must be retrained from scratch.

### Steps

```bash
cd v28_baseline/datasets/fashion_mnist

# 1. Clean old model
rm -rf saved_models/fashion_mnist/
rm -f fashion_mnist_train

# 2. Recompile with fixes
bash compile_fashion_mnist.sh

# 3. Retrain from scratch
./fashion_mnist_train

# 4. Export new model
# (model export is now integrated in training code)

# 5. Verify in notebook
jupyter notebook fashion_mnist_inference.ipynb
```

### Expected Results After Retrain

- **Training accuracy**: ~90-92% (legitimate this time)
- **Test accuracy**: ~90-92% (should match training)
- **Per-class accuracy**: Balanced across all 10 classes
- **No more predicting "Shirt" for everything!**

---

## Lessons Learned

1. **NEVER hardcode dimensions** - always parameterize by input size
2. **Uninitialized memory is dangerous** - can lead to false training signals
3. **Test accuracy != training accuracy** indicates severe overfitting or bugs
4. **One-class prediction** (99% Shirt) is a red flag for broken models
5. **High confidence + wrong predictions** = model learned garbage features

---

## Validation Checklist

After retraining, verify:

- [ ] Test accuracy > 85% (not 10%!)
- [ ] Per-class accuracy balanced (not 0% vs 99%)
- [ ] Predictions distributed across all classes
- [ ] Confusion matrix shows reasonable errors (not all → Shirt)
- [ ] Model file size: fc1_weights.bin = 1152 × 512 × 4 bytes = ~2.4 MB (not 4.2 MB)

---

**Status**: Fixed in code, awaiting retrain
**Affected Models**: Fashion-MNIST only (CIFAR-10/100/SVHN were correct)
**Commit**: 7c03125 - "Fix Fashion-MNIST flatten_size bug"
