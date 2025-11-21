# v28d Streaming Architecture Design

## Executive Summary

**Goal:** Enable training on datasets larger than system RAM by streaming data from disk with zero performance penalty through double buffering.

**Target Use Cases:**
- Medical imaging (100GB-500GB datasets)
- Scientific simulations (1TB+ datasets)
- Video classification (massive frame datasets)
- Research on budget hardware (32GB RAM training on 500GB data)

---

## Problem Statement

### Current Limitation (v28b)

```
Dataset Size ≤ System RAM
```

v28b loads the **entire dataset into managed memory**, which requires:
- Dataset + ~10% overhead must fit in system RAM
- 50GB RAM → ~45GB max dataset
- 128GB RAM → ~115GB max dataset

### Real-World Pain Points

| Domain | Typical Dataset Size | v28b Capable? |
|--------|---------------------|---------------|
| CIFAR-10 | 0.6 GB | ✅ Yes |
| ImageNet-1K | ~150 GB | ⚠️ Needs 170GB+ RAM |
| Medical CT Scans | 200-500 GB | ❌ No (impossible) |
| Video datasets | 500GB - 2TB | ❌ No |
| Scientific HPC | 1TB+ | ❌ No |

---

## Proposed Solution

### Streaming Architecture

**Key Insight:** Training only needs **one batch at a time**. Why load the entire dataset?

```
┌─────────────────────────────────────────────────────────────────┐
│                          DISK (SSD)                             │
│                                                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                │
│  │ Batch 0 │ │ Batch 1 │ │ Batch 2 │ │ Batch 3 │ ...            │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                │
│       │           │           │           │                     │
└───────┼───────────┼───────────┼───────────┼─────────────────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DOUBLE BUFFER (RAM)                        │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │     Buffer A        │    │     Buffer B        │             │
│  │  (Current batch)    │    │  (Next batch)       │             │
│  │                     │    │                     │             │
│  │  Processing on GPU  │    │  Loading from disk  │             │
│  └─────────────────────┘    └─────────────────────┘             │
│           │                          │                          │
│           │ When done, swap          │                          │
│           └──────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GPU (~2GB)                               │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Current Batch (128 × 3072) = 0.5 MB            │            │
│  │  Model Weights = 50 MB                          │            │
│  │  Activations = 200 MB                           │            │
│  │  cuDNN Workspace = 1.7 GB                       │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Buffer A | 1 batch × sizeof(float) | ~0.5 MB for CIFAR-10 |
| Buffer B | 1 batch × sizeof(float) | ~0.5 MB for CIFAR-10 |
| Labels buffer | 2 batches × sizeof(int) | ~1 KB |
| **Total RAM** | **~1-2 MB** | **Regardless of dataset size!** |

---

## Architecture Design

### Module Structure

```
v28d_streaming/
├── common/
│   ├── streaming_loader.cuf      # Core streaming functionality
│   ├── double_buffer.cuf         # Buffer management
│   ├── async_io.cuf              # Asynchronous file I/O
│   └── batch_index_manager.cuf   # Shuffle indices, epoch management
├── datasets/
│   └── cifar10/
│       ├── cifar10_config_stream.cuf  # Streaming config
│       └── cifar10_main_stream.cuf    # Main with streaming
└── docs/
    └── STREAMING_GUIDE.md
```

### Core Components

#### 1. Double Buffer Manager

```fortran
module double_buffer
    use cudafor
    implicit none

    type :: buffer_t
        real(4), managed, allocatable :: data(:,:)    ! (batch_size, features)
        integer, managed, allocatable :: labels(:)     ! (batch_size)
        logical :: ready = .false.
        integer :: batch_idx = -1
    end type buffer_t

    type(buffer_t) :: buffer_a, buffer_b
    type(buffer_t), pointer :: current_buffer, loading_buffer

contains

    subroutine init_double_buffer(batch_size, feature_size)
        integer, intent(in) :: batch_size, feature_size

        allocate(buffer_a%data(batch_size, feature_size))
        allocate(buffer_a%labels(batch_size))
        allocate(buffer_b%data(batch_size, feature_size))
        allocate(buffer_b%labels(batch_size))

        current_buffer => buffer_a
        loading_buffer => buffer_b
    end subroutine

    subroutine swap_buffers()
        type(buffer_t), pointer :: temp
        temp => current_buffer
        current_buffer => loading_buffer
        loading_buffer => temp
    end subroutine

end module double_buffer
```

#### 2. Asynchronous I/O (using OpenMP)

```fortran
module async_io
    use omp_lib
    use double_buffer
    implicit none

    integer :: io_thread_id
    logical :: io_in_progress = .false.

contains

    subroutine start_async_load(batch_idx, file_unit, offset)
        integer, intent(in) :: batch_idx, file_unit
        integer(8), intent(in) :: offset

        !$omp parallel sections num_threads(2)

        !$omp section
        ! I/O Thread: Load next batch
        call load_batch_from_disk(loading_buffer, file_unit, offset)
        loading_buffer%batch_idx = batch_idx
        loading_buffer%ready = .true.

        !$omp section
        ! Main Thread: Returns immediately to continue GPU work

        !$omp end parallel sections
    end subroutine

    subroutine load_batch_from_disk(buf, file_unit, offset)
        type(buffer_t), intent(inout) :: buf
        integer, intent(in) :: file_unit
        integer(8), intent(in) :: offset

        ! Seek to batch position
        read(file_unit, pos=offset) buf%data
        buf%ready = .true.
    end subroutine

end module async_io
```

#### 3. Streaming Data Loader

```fortran
module streaming_loader
    use double_buffer
    use async_io
    implicit none

    ! File handles
    integer :: data_file_unit = 20
    integer :: label_file_unit = 21

    ! Dataset info
    integer(8) :: total_samples
    integer :: batch_size
    integer :: feature_size
    integer(8) :: bytes_per_sample

    ! Shuffle indices
    integer, allocatable :: shuffle_indices(:)

contains

    subroutine init_streaming(filename, n_samples, batch_sz, feat_sz)
        character(len=*), intent(in) :: filename
        integer(8), intent(in) :: n_samples
        integer, intent(in) :: batch_sz, feat_sz

        total_samples = n_samples
        batch_size = batch_sz
        feature_size = feat_sz
        bytes_per_sample = feat_sz * 4  ! float32

        ! Open file for streaming
        open(unit=data_file_unit, file=filename, &
             form='unformatted', access='stream', status='old')

        ! Initialize double buffer
        call init_double_buffer(batch_sz, feat_sz)

        ! Pre-load first batch
        call load_batch_sync(0)
        current_buffer%ready = .true.

        ! Start loading second batch asynchronously
        call start_async_load(1)

    end subroutine

    subroutine get_next_batch(batch_data, batch_labels)
        real(4), intent(out) :: batch_data(:,:)
        integer, intent(out) :: batch_labels(:)

        ! Wait for current buffer to be ready
        do while (.not. current_buffer%ready)
            ! Spin wait (could add sleep for efficiency)
        end do

        ! Copy to output (or return pointer for zero-copy)
        batch_data = current_buffer%data
        batch_labels = current_buffer%labels

        ! Swap buffers
        call swap_buffers()

        ! Start loading next batch asynchronously
        call start_async_load(current_buffer%batch_idx + 2)

    end subroutine

end module streaming_loader
```

---

## Performance Analysis

### Timing Budget

| Operation | Time | Notes |
|-----------|------|-------|
| **GPU Batch Processing** | ~50 ms | Forward + backward + update |
| **SSD Read (NVMe)** | ~0.1 ms | 0.5 MB @ 5 GB/s |
| **SSD Read (SATA)** | ~1 ms | 0.5 MB @ 500 MB/s |
| **HDD Read** | ~5 ms | 0.5 MB @ 100 MB/s |

### Overlap Analysis

```
Time →

GPU:   [====Batch 0====][====Batch 1====][====Batch 2====]
         50ms              50ms              50ms

I/O:   [B1][B2][B3]...
       0.1ms each (hidden behind GPU work)

Result: Zero I/O overhead with any SSD!
```

### When Streaming Helps vs Hurts

| Scenario | v28b (Load All) | v28d (Streaming) | Winner |
|----------|-----------------|------------------|--------|
| Dataset < RAM | Fast load, fast train | Slight overhead | v28b |
| Dataset = RAM | May swap, unstable | Smooth, stable | v28d |
| Dataset > RAM | **IMPOSSIBLE** | Works perfectly | v28d |
| Dataset >> RAM | **IMPOSSIBLE** | Works perfectly | v28d |

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

1. **Double buffer module**
   - Allocate two batch-sized buffers
   - Swap mechanism
   - Ready flags

2. **Synchronous streaming**
   - Open file, seek to position
   - Read single batch
   - Test correctness

### Phase 2: Async I/O (Week 2)

3. **OpenMP threading**
   - Separate I/O thread
   - Non-blocking reads
   - Synchronization primitives

4. **Overlap verification**
   - Measure I/O time vs GPU time
   - Verify zero overhead with SSD

### Phase 3: Integration (Week 3)

5. **Training loop integration**
   - Replace batch extraction with streaming
   - Handle epoch boundaries
   - Implement shuffling for streaming

6. **Testing and benchmarks**
   - Correctness tests (compare to v28b)
   - Performance benchmarks
   - Memory usage verification

### Phase 4: Polish (Week 4)

7. **Error handling**
   - File not found
   - Partial reads
   - Disk full scenarios

8. **Documentation**
   - User guide
   - Performance tuning guide
   - Troubleshooting

---

## Configuration Options

### Proposed Config Interface

```fortran
module streaming_config
    implicit none

    ! Memory modes
    integer, parameter :: MODE_MANAGED = 1    ! v28b: Load all into RAM
    integer, parameter :: MODE_STREAMING = 2  ! v28d: Stream from disk
    integer, parameter :: MODE_MMAP = 3       ! Future: Memory-mapped file

    ! User-configurable
    integer :: memory_mode = MODE_MANAGED     ! Default: current behavior
    integer :: prefetch_batches = 2           ! Number of batches to prefetch
    logical :: enable_shuffle = .true.        ! Shuffle indices each epoch

    ! Auto-detected
    integer(8) :: available_ram               ! System RAM
    integer(8) :: dataset_size                ! From file size

contains

    subroutine auto_select_mode()
        ! Automatically choose best mode based on resources
        if (dataset_size < available_ram * 0.8) then
            memory_mode = MODE_MANAGED  ! Dataset fits, use fast path
            print *, "Auto-selected: MANAGED mode (dataset fits in RAM)"
        else
            memory_mode = MODE_STREAMING  ! Must stream
            print *, "Auto-selected: STREAMING mode (dataset > RAM)"
        endif
    end subroutine

end module streaming_config
```

### Command-Line Control

```bash
# Force streaming mode (for testing)
./cifar10_train --memory-mode=streaming

# Force managed mode (if dataset fits)
./cifar10_train --memory-mode=managed

# Auto-detect (default)
./cifar10_train --memory-mode=auto
```

---

## Shuffling Strategy for Streaming

### Challenge

With streaming, we can't shuffle the data array directly (it's on disk). We need to shuffle **access order** instead.

### Solution: Index-Based Shuffling

```fortran
! At epoch start:
1. Generate shuffle_indices = [0, 1, 2, ..., N-1]
2. Fisher-Yates shuffle the indices
3. When loading batch i, read samples at indices[i*batch_size : (i+1)*batch_size]

! Problem: Random disk access is slow!
```

### Optimization: Block Shuffling

```fortran
! Divide dataset into blocks (e.g., 1000 batches per block)
1. Shuffle block order: [Block 5, Block 2, Block 8, ...]
2. Within each block, shuffle batch order
3. Sequential reads within each block!

! Result: Good randomization with mostly sequential I/O
```

### Shuffle Modes

| Mode | Randomization | I/O Pattern | Performance |
|------|---------------|-------------|-------------|
| None | ❌ | Sequential | Fastest |
| Block | ✅ Good | Mostly sequential | Fast |
| Full | ✅ Best | Random | Slower (SSD OK) |

---

## File Format Considerations

### Current Format (v28b)

```
images_train.bin: [sample0_data][sample1_data]...[sampleN_data]
                  Contiguous, good for full load
```

### Streaming-Optimized Format

**Option A: Keep Current Format**
- Works fine with sequential access
- Random access requires seek per sample
- OK for NVMe, slow for HDD

**Option B: Batch-Aligned Format**
```
images_train.bin: [batch0_header][batch0_data][batch1_header][batch1_data]...
                  Each batch is contiguous
                  Easy to load any batch with single read
```

**Option C: Chunked Format**
```
train_chunk_000.bin: batches 0-99
train_chunk_001.bin: batches 100-199
...
Benefits: Can memory-map individual chunks
```

**Recommendation:** Start with Option A, optimize if needed.

---

## Memory-Mapped Files (Future: v28e)

### Concept

Let the OS handle paging automatically:

```fortran
! Instead of manual streaming:
call mmap(filename, gpu_train_data)

! OS pages data in/out automatically
! Works like managed memory but backed by file
```

### Pros
- Simplest implementation
- OS optimizes paging
- Works with any access pattern

### Cons
- Less control over timing
- May page out at wrong times
- Platform-specific implementations

### When to Use
- Datasets 2-10× RAM size
- Random access patterns
- Prototyping before full streaming impl

---

## Benchmarking Plan

### Metrics to Measure

1. **Training throughput** (samples/second)
2. **Memory usage** (peak RAM, GPU)
3. **I/O wait time** (time spent waiting for data)
4. **End-to-end time** (total training time)

### Test Cases

| Test | Dataset Size | RAM | Expected Result |
|------|--------------|-----|-----------------|
| Small | 1 GB | 64 GB | Streaming ~= Managed |
| Medium | 50 GB | 64 GB | Streaming slightly slower |
| Large | 100 GB | 64 GB | Managed fails, Streaming works |
| Huge | 500 GB | 64 GB | Only Streaming works |

### Baseline Comparisons

- v28b managed memory (where possible)
- PyTorch DataLoader with num_workers
- TensorFlow tf.data pipeline

---

## Risk Analysis

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| I/O bottleneck on HDD | Medium | High | Require SSD, or larger prefetch |
| OpenMP complexity | Medium | Medium | Start with synchronous version |
| File format incompatibility | Low | Low | Support both formats |
| Race conditions | Medium | High | Careful synchronization, testing |

### Performance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Overhead > benefit | Low | High | Only use when dataset > RAM |
| Shuffle hurts I/O | Medium | Medium | Block shuffling option |
| GPU starvation | Low | High | Prefetch more batches |

---

## Success Criteria

### MVP (Minimum Viable Product)

- [ ] Stream from disk without loading full dataset
- [ ] Double buffering hides I/O latency
- [ ] Correctness matches v28b (same accuracy)
- [ ] Works with NVMe SSD

### Production Ready

- [ ] Automatic mode selection
- [ ] Block shuffling implemented
- [ ] Error handling complete
- [ ] Documentation complete
- [ ] Benchmarks show <5% overhead vs v28b (where comparable)

### Stretch Goals

- [ ] Memory-mapped file support
- [ ] Multiple prefetch threads
- [ ] Compression support (load compressed, decompress on GPU)
- [ ] Multi-file dataset support

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Double buffer + synchronous streaming |
| 2 | Async I/O with OpenMP |
| 3 | Training integration + shuffling |
| 4 | Testing, benchmarks, documentation |
| 5 | Polish, edge cases, release |

---

## Conclusion

v28d streaming architecture will enable training on **datasets of any size** with minimal RAM requirements. Combined with v28b managed memory (for datasets that fit in RAM) and v28c warp shuffle (for faster GPU operations), this creates a complete solution for scientific computing on budget hardware.

**Key benefits:**
- Train on 500GB+ datasets with 32GB RAM
- Zero performance penalty with SSD
- Automatic mode selection for optimal performance
- Enables medical imaging, video, and scientific HPC workloads

**Next step:** Implement Phase 1 (double buffer + sync streaming) as proof of concept.
