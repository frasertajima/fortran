# v28 Baseline Validation Checklist

**Critical sanity checks to run after training any dataset**

---

## ✅ Pre-Training Checks

### 1. Verify Parameterized Flatten Size

```bash
# Check that flatten_size is parameterized (NOT hardcoded)
grep -n "CONV3_FILTERS \* 4 \* 4" datasets/<dataset>/<dataset>_main.cuf

# Should return: NO MATCHES
# If matches found: flatten_size is hardcoded (BUG!)
```

**Expected**: Zero matches - all allocations should use:
```fortran
CONV3_FILTERS * ((INPUT_HEIGHT/4)/2) * ((INPUT_WIDTH/4)/2)
```

### 2. Verify Model Export Integration

```bash
# Check compilation script includes model_export
grep "model_export.cuf" datasets/<dataset>/compile_*.sh

# Check main file imports model_export
grep "use model_export" datasets/<dataset>/<dataset>_main.cuf

# Check export calls exist
grep "call export_model_generic" datasets/<dataset>/<dataset>_main.cuf
```

**Expected**: All three commands return matches

---

## ✅ Post-Training Checks

### 3. Verify Model Export Success

```bash
# Check saved_models directory exists
ls -la datasets/<dataset>/saved_models/<dataset>/

# Expected: 20 files
#   - 1 model_metadata.txt
#   - 19 binary files (.bin)
```

**Expected file list**:
```
model_metadata.txt
conv1_weights.bin, conv1_bias.bin
conv2_weights.bin, conv2_bias.bin
conv3_weights.bin, conv3_bias.bin
fc1_weights.bin, fc1_bias.bin
fc2_weights.bin, fc2_bias.bin
fc3_weights.bin, fc3_bias.bin
bn1_scale.bin, bn1_bias.bin, bn1_running_mean.bin, bn1_running_var.bin
bn2_scale.bin, bn2_bias.bin, bn2_running_mean.bin, bn2_running_var.bin
bn3_scale.bin, bn3_bias.bin, bn3_running_mean.bin, bn3_running_var.bin
```

### 4. Verify Model File Sizes

```bash
# Check fc1_weights.bin size matches expected flatten_size
ls -lh datasets/<dataset>/saved_models/<dataset>/fc1_weights.bin
```

**Expected sizes**:
- **CIFAR-10/100/SVHN** (32×32): fc1_weights = 512 × 2048 × 4 bytes = **4.2 MB**
- **Fashion-MNIST** (28×28): fc1_weights = 512 × 1152 × 4 bytes = **2.4 MB**

**Red flag**: If Fashion-MNIST fc1_weights.bin is 4.2 MB → trained with bug!

### 5. Verify Metadata Content

```bash
cat datasets/<dataset>/saved_models/<dataset>/model_metadata.txt
```

**Expected content**:
```
======================================================================
v28 Baseline - Trained Model Export
======================================================================
Dataset: <DATASET-NAME>
Test Accuracy: XX.XX%
Epochs Trained: XX
Export Date: 2025-11-17

Files Exported:
  Convolutional Layers:
    - conv1_weights.bin, conv1_bias.bin
    ...
  BatchNorm Layers (Conv blocks only):
    - bn[1-3]_scale.bin, bn[1-3]_bias.bin
    - bn[1-3]_running_mean.bin, bn[1-3]_running_var.bin

Load with: v28_baseline/inference/model_loader.py
======================================================================
```

---

## ✅ Inference Validation

### 6. Load Model in Python

```python
import sys
sys.path.append('../../inference')
from model_loader import load_v28_model

# For CIFAR-10/100/SVHN (32×32 RGB)
model = load_v28_model('./saved_models/cifar10/',
                       in_channels=3, num_classes=10, input_size=32)

# For Fashion-MNIST (28×28 grayscale)
model = load_v28_model('./saved_models/fashion_mnist/',
                       in_channels=1, num_classes=10, input_size=28)

print("✅ Model loaded successfully!")
```

**Expected output**:
```
Loading v28 model from: saved_models/<dataset>
Auto-detected flatten_size from fc1_weights: 2048  (or 1152 for Fashion-MNIST)
Loading Conv1...
Loading Conv2...
Loading Conv3...
Loading FC1...
Loading FC2...
Loading FC3...
Loading BatchNorm layers...
✅ Model loaded successfully!
```

**Red flags**:
- `ValueError: Shape mismatch` → Model file doesn't match expected size
- Auto-detected flatten_size wrong for dataset

### 7. Test Inference Accuracy

```python
# Run inference notebook completely
jupyter notebook datasets/<dataset>/<dataset>_inference.ipynb
```

**Expected results**:
- **Test Accuracy ≥ 85%** for all datasets
- **Per-class accuracy balanced** (not 99% one class, 0% others)
- **Predictions distributed** across all classes
- **Confidence reasonable** (not 99.99% for all predictions)

**Red flags indicating broken model**:
- Test accuracy < 50% (worse than baseline)
- Predicts one class 90%+ of the time (like Fashion-MNIST "Shirt" bug)
- High confidence + wrong predictions (learned garbage features)
- Per-class accuracy: some 0%, others 90%+

### 8. Verify Architecture Consistency

```python
# Check model architecture matches v28 spec
print(model)
```

**Expected architecture**:
```
V28_CNN(
  (conv1): Conv2d(in_channels, 32, kernel_size=3, padding=1)
  (bn1): BatchNorm2d(32)
  (pool1): MaxPool2d(kernel_size=2, stride=2)
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)
  (bn2): BatchNorm2d(64)
  (pool2): MaxPool2d(kernel_size=2, stride=2)
  (conv3): Conv2d(64, 128, kernel_size=3, padding=1)
  (bn3): BatchNorm2d(128)
  (pool3): MaxPool2d(kernel_size=2, stride=2)
  (fc1): Linear(flatten_size, 512)
  (dropout1): Dropout(p=0.5)
  (fc2): Linear(512, 256)
  (dropout2): Dropout(p=0.5)
  (fc3): Linear(256, num_classes)
  (leaky_relu): LeakyReLU(negative_slope=0.01)
)
```

**Key points**:
- 3 convolutional blocks with BatchNorm
- NO BatchNorm after FC layers
- LeakyReLU activation (not ELU)
- Dropout 0.5 (not 0.3)

---

## ✅ Dataset-Specific Sanity Checks

### Fashion-MNIST (28×28 grayscale)

- [ ] fc1_weights.bin = **2.4 MB** (NOT 4.2 MB)
- [ ] Auto-detected flatten_size = **1152** (NOT 2048)
- [ ] Test accuracy > 85%
- [ ] NOT predicting "Shirt" 99% of the time

### CIFAR-10 (32×32 RGB)

- [ ] fc1_weights.bin = **4.2 MB**
- [ ] Auto-detected flatten_size = **2048**
- [ ] Test accuracy > 75%
- [ ] NOT predicting "cat" 70%+ of the time

### CIFAR-100 (32×32 RGB)

- [ ] fc1_weights.bin = **4.2 MB**
- [ ] Auto-detected flatten_size = **2048**
- [ ] Test accuracy > 50% (harder dataset)
- [ ] Predictions distributed across many classes

### SVHN (32×32 RGB digits)

- [ ] fc1_weights.bin = **4.2 MB**
- [ ] Auto-detected flatten_size = **2048**
- [ ] Test accuracy > 85%
- [ ] Predictions distributed across digits 0-9

---

## 🚨 Common Failure Modes

### 1. Hardcoded Flatten Size Bug

**Symptoms**:
- Fashion-MNIST fc1_weights.bin is 4.2 MB (should be 2.4 MB)
- Test accuracy ~10% (predicts one class for everything)
- Model "learned" random memory garbage

**Fix**: Re-parameterize and retrain

### 2. Missing Model Export

**Symptoms**:
- saved_models/ directory doesn't exist after training
- No .bin files generated

**Fix**: Check compile script includes model_export.cuf, main file has export calls

### 3. Broken Model (Overfit to Noise)

**Symptoms**:
- Training accuracy 90%+, test accuracy <50%
- Predicts one class 90%+ of the time
- High confidence + wrong predictions

**Fix**: Check for bugs in training (uninitialized memory, wrong flatten_size)

### 4. Architecture Mismatch

**Symptoms**:
- `ValueError: Shape mismatch` when loading
- Model loads but gives random predictions
- Flatten_size detected incorrectly

**Fix**: Ensure model_loader.py matches actual v28 architecture

---

## 📋 Quick Validation Script

```bash
#!/bin/bash
# Run this after training any dataset

DATASET=$1  # cifar10, cifar100, svhn, fashion_mnist

echo "=== Validating $DATASET ==="

# 1. Check files exist
if [ ! -d "datasets/$DATASET/saved_models/$DATASET" ]; then
    echo "❌ ERROR: saved_models directory not found!"
    exit 1
fi

FILE_COUNT=$(ls datasets/$DATASET/saved_models/$DATASET/*.bin | wc -l)
if [ "$FILE_COUNT" != "19" ]; then
    echo "❌ ERROR: Expected 19 .bin files, found $FILE_COUNT"
    exit 1
fi

# 2. Check fc1_weights size
FC1_SIZE=$(stat -f%z datasets/$DATASET/saved_models/$DATASET/fc1_weights.bin 2>/dev/null || stat -c%s datasets/$DATASET/saved_models/$DATASET/fc1_weights.bin)

if [ "$DATASET" == "fashion_mnist" ]; then
    EXPECTED_SIZE=2359296  # 512*1152*4
    if [ "$FC1_SIZE" != "$EXPECTED_SIZE" ]; then
        echo "❌ ERROR: fc1_weights wrong size! Expected 2.4MB, got $(($FC1_SIZE/1024/1024))MB"
        echo "   Model was trained with flatten_size bug!"
        exit 1
    fi
else
    EXPECTED_SIZE=4194304  # 512*2048*4
    if [ "$FC1_SIZE" != "$EXPECTED_SIZE" ]; then
        echo "❌ ERROR: fc1_weights wrong size! Expected 4.2MB, got $(($FC1_SIZE/1024/1024))MB"
        exit 1
    fi
fi

echo "✅ Model export validation passed!"
echo "   19 binary files present"
echo "   fc1_weights.bin correct size"
echo ""
echo "Next: Run inference notebook to validate accuracy"
```

Save as `validate_model.sh`, then run:
```bash
bash validate_model.sh cifar10
bash validate_model.sh fashion_mnist
```

---

**Status**: Mandatory baseline validation
**Last Updated**: 2025-11-17
**Applies To**: All v28 datasets
