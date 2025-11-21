# Streaming Data Format Requirements

## The Critical Insight

**Streaming mode requires SAMPLE-MAJOR data layout, not feature-major.**

This is the single most important thing to understand about the streaming architecture. Getting this wrong results in training that appears to work but produces ~10-12% accuracy instead of the expected ~77-80%.

## Why This Matters

### The Problem

Most dataset preparation tools (including CIFAR-10's standard binary format) store data in **feature-major** order:

```
Feature-Major Layout (WRONG for streaming):
┌─────────────────────────────────────────────────────────────┐
│ f0_s0 f0_s1 f0_s2 ... f0_sN │ f1_s0 f1_s1 ... │ f2_s0 ... │
└─────────────────────────────────────────────────────────────┘
  ← all samples, feature 0 →   ← feature 1 →     ← etc →

Memory layout: [feature 0 of ALL samples] [feature 1 of ALL samples] ...
```

This works fine when you load the entire dataset into RAM and reshape it. Fortran's column-major order naturally handles the transpose when reading into `(N_samples, N_features)`.

### The Streaming Problem

When streaming batch-by-batch, you read **contiguous chunks** from disk:

```fortran
! Read bytes for batch starting at sample idx
read(unit, pos=idx * feature_size * 4 + 1) batch_data(:, 1:batch_size)
```

With feature-major data, this reads:
- Sample 0, features 0-127 (if batch_size=128 and we start at position 0)
- **NOT** samples 0-127, all features!

The result: each "sample" in your batch is actually a slice across multiple real samples, causing the model to learn garbage patterns.

### The Solution

Store data in **sample-major** order:

```
Sample-Major Layout (CORRECT for streaming):
┌─────────────────────────────────────────────────────────────┐
│ s0_f0 s0_f1 s0_f2 ... s0_fM │ s1_f0 s1_f1 ... │ s2_f0 ... │
└─────────────────────────────────────────────────────────────┘
  ← all features, sample 0 →   ← sample 1 →      ← etc →

Memory layout: [all features of sample 0] [all features of sample 1] ...
```

Now sequential reads give you complete samples:

```
Read 128 samples × 3072 features = 393,216 floats
→ Gets samples 0-127, each with all 3072 features intact
```

## Fortran Memory Layout

Understanding Fortran's column-major order is essential:

```fortran
real :: data(3072, 50000)  ! (features, samples)
```

In memory, this is stored as:
```
data(1,1), data(2,1), data(3,1), ..., data(3072,1),  ! Sample 1, all features
data(1,2), data(2,2), data(3,2), ..., data(3072,2),  ! Sample 2, all features
...
```

**This IS sample-major in memory!** So we need our binary file to match this layout.

## Conversion Workflow

### Step 1: Check Your Current Format

```python
import numpy as np

# Load your data file
data = np.fromfile('images_train.bin', dtype=np.float32)
print(f"Total elements: {len(data)}")

# For CIFAR-10: 50000 samples × 3072 features = 153,600,000
# Check if it's feature-major or sample-major

# Read first few values and compare with known sample 0
# Feature-major: first 50000 values are feature 0 of all samples
# Sample-major: first 3072 values are all features of sample 0
```

### Step 2: Convert to Sample-Major

```python
import numpy as np

# Parameters
N_SAMPLES = 50000
N_FEATURES = 3072

# Load feature-major data
data_fm = np.fromfile('images_train.bin', dtype=np.float32)
data_fm = data_fm.reshape(N_FEATURES, N_SAMPLES)  # (features, samples)

# Convert to sample-major (transpose)
data_sm = data_fm.T  # Now (samples, features)

# Verify: data_sm[i, :] should be all features of sample i
# For C-contiguous (row-major) numpy array, .T.flatten() gives sample-major

# Save as sample-major binary (C-contiguous = row-major = sample-major)
data_sm.astype(np.float32).tofile('images_train_streaming.bin')
```

### Step 3: Verify the Conversion

```python
# Quick verification
original = np.fromfile('images_train.bin', dtype=np.float32)
converted = np.fromfile('images_train_streaming.bin', dtype=np.float32)

# In sample-major, sample 0's features are at positions 0:3072
sample_0_sm = converted[0:3072]

# In feature-major, sample 0's features are at positions 0, 50000, 100000, ...
sample_0_fm = original[0::50000][:3072]

# These should match!
print(f"Match: {np.allclose(sample_0_sm, sample_0_fm)}")
```

## Directory Structure and File Naming

**Files use a `_stream` suffix to prevent accidental format mixing.**

This is critical: using the wrong format silently produces ~10% accuracy instead of ~77%.

```
datasets/cifar10/
├── cifar10_data/                    # Original format (full RAM mode)
│   ├── images_train.bin             # Feature-major
│   ├── labels_train.bin
│   ├── images_test.bin
│   └── labels_test.bin
├── cifar10_data_streaming/          # Streaming format
│   ├── images_train_stream.bin      # Sample-major (note _stream suffix)
│   ├── labels_train_stream.bin
│   ├── images_test_stream.bin
│   └── labels_test_stream.bin
└── cifar10_train                    # Executable
```

The `_stream` suffix serves as a safety check - if you accidentally point the streaming
loader at `images_train.bin` instead of `images_train_stream.bin`, you'll get a
file-not-found error rather than silent data corruption.

## The Streaming Buffer

In `streaming_data_loader.cuf`, the buffer is shaped to match sample-major reads:

```fortran
type :: stream_buffer_t
    real(4), managed, allocatable :: data(:,:)    ! (feature_size, batch_size)
    integer, managed, allocatable :: labels(:)     ! (batch_size)
end type
```

Reading into `data(:, 1:n)` from a sample-major file:
- Reads `n × feature_size` contiguous floats
- Fortran column-major fills: `data(1,1), data(2,1), ..., data(feature_size,1), data(1,2), ...`
- Result: column `i` contains all features of sample `i`

## Common Mistakes

### 1. Wrong reshape order

```python
# WRONG: numpy default is C-order (row-major)
data.reshape(50000, 3072)  # This transposes when read by Fortran!

# RIGHT: explicitly use Fortran order or transpose
data.reshape(3072, 50000).T  # Explicit sample-major
```

### 2. Not accounting for Fortran column-major

```fortran
! WRONG: Reading into (samples, features) from sample-major file
real :: batch_data(batch_size, feature_size)
read(unit, pos=pos) batch_data  ! Columns fill first = wrong!

! RIGHT: Reading into (features, samples)
real :: batch_data(feature_size, batch_size)
read(unit, pos=pos) batch_data  ! Sample data fills column by column
```

### 3. Forgetting the transpose to network format

The CNN expects `(samples, features)` but we read into `(features, samples)`:

```fortran
! After streaming read: stream_data is (features, samples)
! Network expects: batch_data is (samples, features)
batch_data(1:n, :) = transpose(stream_data(:, 1:n))
```

## Quick Reference

| Format | File Layout | Fortran Array Shape | Use Case |
|--------|-------------|---------------------|----------|
| Feature-major | `[f0_s0, f0_s1, ...]` | `(samples, features)` | Full RAM mode |
| Sample-major | `[s0_f0, s0_f1, ...]` | `(features, samples)` | Streaming mode |

## Testing Your Setup

Before running full training, test with a small verification:

```fortran
! Read one sample from streaming file
read(unit, pos=1) test_sample(1:feature_size)

! Compare with known ground truth from original data
! They should match exactly
```

Or use the included test programs in `v28d_streaming/tests/`.

---

**Key Takeaway**: Streaming requires sample-major binary files. Convert once, stream forever.
