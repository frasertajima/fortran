# CIFAR-10 cuDNN - v28d Streaming Architecture

**High-performance, modular CNN training framework in CUDA Fortran with streaming data loading for datasets of ANY size (limited only by SSD storage)**

## What's New in v28d

**v28d adds streaming data loading** - train on datasets larger than GPU RAM AND system RAM combined!

| Feature | v28c | v28d |
|---------|------|------|
| Max dataset size | Limited by RAM | **Unlimited** |
| 32GB dataset | Would OOM | **Works (1.35 GB GPU)** |
| Memory usage | Scales with dataset | **Constant (~3 MB)** |
| I/O overhead | N/A | **Zero (hidden)** |

### Validated Results

```
32GB Dataset Test:
  GPU memory used:     1.35 GB (constant)
  Memory increase:     0.00 MB
  Throughput:          1,273 MB/s
  MEMORY BOUNDED:      PASS
```

## Key Features

- **5x faster than PyTorch** (28.9s vs 146s on CIFAR-10) for full RAM version (automatically handles up to main RAM, not GPU RAM datasets)
- Streaming version takes 52.5s on CIFAR-10 due to overhead (PyTorch crashes on OOM datasets); run with streaming format dataset and --stream flag
- **Train on ANY dataset size** - 1GB, 100GB, 1TB - same memory footprint
- **Double-buffered streaming** - I/O completely hidden behind GPU work
- **OpenMP async I/O** - Disk reads overlap with GPU computation
- **Block shuffling** - Good randomization with efficient I/O
- **Fully modular** - All datasets use the same streaming infrastructure

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          DISK (SSD)                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                │
│  │ Batch 0 │ │ Batch 1 │ │ Batch 2 │ │ Batch 3 │ ...            │
└───────┼───────────┼───────────┼───────────┼─────────────────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DOUBLE BUFFER (RAM)                        │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │     Buffer A        │    │     Buffer B        │             │
│  │  (GPU processing)   │◄──►│  (Loading from disk)│             │
│  └─────────────────────┘    └─────────────────────┘             │
│           │                          │                          │
│           └────── SWAP ──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GPU (Training)                           │
│  Current Batch → Forward → Backward → Update                    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Navigate to dataset
cd v28d_streaming/datasets/cifar10

# Prepare dataset (creates binary files)
python prepare_cifar10.py

# Convert to streaming format (creates *_stream.bin files)
python ../../tools/convert_to_streaming.py \
    --input-dir cifar10_data/ \
    --output-dir cifar10_data_streaming/ \
    --preset cifar10

# Compile with streaming support
bash compile_cifar10.sh

# Train in streaming mode
./cifar10_train --stream
```

## CRITICAL: Data Format Requirements

**Streaming mode requires SAMPLE-MAJOR data layout.**

Standard ML datasets (including CIFAR-10) are typically stored in FEATURE-MAJOR format.
Using the wrong format silently produces ~10% accuracy instead of ~77%.

| Format | Layout | Use Case |
|--------|--------|----------|
| Feature-major | `[f0_s0, f0_s1, ... f1_s0, f1_s1, ...]` | Full RAM mode |
| Sample-major | `[s0_f0, s0_f1, ... s1_f0, s1_f1, ...]` | **Streaming mode** |

**Safety convention**: Streaming files use `_stream` suffix to prevent mixing:
- `images_train.bin` - Original feature-major (full RAM)
- `images_train_stream.bin` - Converted sample-major (streaming)

See `docs/STREAMING_DATA_FORMAT.md` for detailed explanation.

## Streaming Module Usage

```fortran
use streaming_data_loader

! Initialize streaming
call streaming_init(data_file, label_file, num_samples, feature_size, batch_size)

! Set shuffle mode (optional)
call streaming_set_shuffle_mode(SHUFFLE_BLOCK, block_size=50)

! Training loop
do epoch = 1, num_epochs
    call streaming_start_epoch()  ! Reshuffles indices
    
    do while (.true.)
        call streaming_get_batch(batch_data, batch_labels, actual_size)
        if (actual_size == 0) exit  ! End of epoch
        
        ! Train on batch (GPU work happens while next batch loads)
        call forward_pass(batch_data, ...)
        call backward_pass(...)
    end do
end do

! Cleanup
call streaming_cleanup()
```

## Shuffle Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `SHUFFLE_NONE` | Sequential access | Debugging, deterministic runs |
| `SHUFFLE_BLOCK` | Shuffle blocks of batches | **Default** - good balance |
| `SHUFFLE_FULL` | Full random shuffle | NVMe SSD with fast random access |

## Performance

### Throughput (NVMe SSD)
- Sequential read: 8,337 MB/s
- Random batch read: 0.18 ms per batch
- Streaming throughput: 1,273 MB/s sustained

### Memory Usage
| Dataset Size | GPU Memory | System RAM |
|--------------|------------|------------|
| 1 GB | 1.35 GB | ~3 MB |
| 32 GB | 1.35 GB | ~3 MB |
| 500 GB | 1.35 GB | ~3 MB |
| 1 TB | 1.35 GB | ~3 MB |

## Supported Datasets

| Dataset | Size | Status |
|---------|------|--------|
| CIFAR-10 | 0.6 GB | Ready |
| CIFAR-100 | 0.6 GB | Ready |
| SVHN | 0.9 GB | Ready |
| Fashion-MNIST | 0.5 GB | Ready |
| ImageNet-1K | 150 GB | Enabled |
| Custom | Any | Enabled |

## Preparing Large Datasets (Direct Streaming Format)

For datasets too large to fit in memory, use the template to write directly in streaming format:

```bash
# Copy the template
cp tools/prepare_streaming_template.py my_dataset/prepare_streaming.py

# Edit the template:
# 1. Set N_TRAIN_SAMPLES, N_TEST_SAMPLES, N_FEATURES
# 2. Implement load_and_process_sample() for your data source
# 3. Run it
python my_dataset/prepare_streaming.py
```

The template supports loading from:
- Individual image files (JPG, PNG)
- HDF5 datasets (memory-mapped)
- Database queries
- Network streams
- Any source that can yield one sample at a time

Memory usage is **constant** regardless of dataset size - samples are written one at a time.

## Converting Existing Datasets

For datasets already in feature-major format:

```bash
# Using presets for known datasets
python tools/convert_to_streaming.py \
    --input-dir mydata/ \
    --output-dir mydata_streaming/ \
    --preset cifar10

# Custom dimensions
python tools/convert_to_streaming.py \
    --input-dir mydata/ \
    --output-dir mydata_streaming/ \
    --samples-train 100000 \
    --samples-test 10000 \
    --features 2048

# List available presets
python tools/convert_to_streaming.py --list-presets
```

The converter automatically adds `_stream` suffix to output files.

## Test Suite

The streaming architecture was validated with 15 standalone tests:

```bash
cd ../v28d_streaming_tests

# Build all tests
make all

# Run full test suite
./scripts/run_all_tests.sh

# Run 32GB stress test
OMP_NUM_THREADS=2 ./phase5_stress_tests/test_15_16gb_memory_validation
```

See `v28d_streaming_tests/RESULTS.md` for detailed test results.

## Files

```
v28d_streaming/
├── common/
│   ├── streaming_data_loader.cuf   # Streaming module
│   ├── gpu_batch_extraction.cuf    # GPU batch extraction
│   ├── warp_shuffle.cuf            # Warp shuffle reductions
│   ├── adam_optimizer.cuf          # Adam optimizer
│   └── ...
├── datasets/
│   ├── cifar10/
│   │   ├── cifar10_data/           # Original feature-major files
│   │   └── cifar10_data_streaming/ # Sample-major *_stream.bin files
│   ├── cifar100/
│   ├── svhn/
│   └── fashion_mnist/
├── tools/
│   ├── convert_to_streaming.py     # Convert existing feature-major files
│   └── prepare_streaming_template.py  # Template for direct streaming output
├── docs/
│   └── STREAMING_DATA_FORMAT.md    # Detailed format documentation
└── README.md
```

## Comparison with Previous Versions

| Version | Max Dataset | Key Feature |
|---------|-------------|-------------|
| v28 baseline | GPU RAM | Modular architecture |
| v28b managed | System RAM | Managed memory |
| v28c warp shuffle | System RAM | GPU metrics |
| **v28d streaming** | **Unlimited** | **Streaming I/O** |

## Requirements

- NVIDIA GPU with CUDA support
- nvfortran compiler with OpenMP (`-mp` flag)
- NVMe SSD recommended (SATA SSD works, HDD not recommended)
- 2+ CPU threads for async I/O

## Compilation Flags

```bash
nvfortran -cuda -mp -O3 -o train program.cuf
```

The `-mp` flag enables OpenMP for async I/O threading.

---

**Repository**: https://github.com/frasertajima/fortran/CIFAR-10_cuDNN

**Version**: v28d Streaming Architecture

**Status**: Production-ready

**Last Updated**: 2025-11-21
