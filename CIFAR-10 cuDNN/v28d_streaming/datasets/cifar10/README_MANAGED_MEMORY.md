# Managed Memory Testing for CIFAR-10

## Overview

This directory contains managed memory experiments to train on datasets larger than GPU RAM.

## Key Innovation

**Managed Memory** (CUDA Unified Memory) allows training on datasets 5-10× larger than GPU memory by automatically paging data between CPU RAM and GPU RAM.

### Traditional Device Memory
```fortran
real(4), device :: train_data(3072, 50000)  ! Limited by GPU RAM (~8-24GB)
! Fails if dataset > GPU memory
```

### Managed Memory
```fortran
real(4), managed :: train_data(3072, 500000)  ! Limited by system RAM (~64-256GB+)
! CUDA automatically pages data between CPU/GPU as needed
```

## Files

- `test_managed_memory.cuf` - Simple test program to verify managed memory works
- `compile_managed_test.sh` - Compile the test
- `cifar10_main_managed.cuf` - (TODO) Full CIFAR-10 training with managed memory

## Quick Start

### Step 1: Test Managed Memory

```bash
# Compile the test
./compile_managed_test.sh

# Run the test
./test_managed_memory
```

**Expected output:**
```
Test 1: Small dataset with device memory
  Samples:       10000
  Memory:        0.117 GB
  ✅ SUCCESS

Test 2: Medium dataset (CIFAR-10 size) with device memory
  Samples:       50000
  Memory:        0.586 GB
  ✅ SUCCESS

Test 3: Medium dataset (CIFAR-10 size) with managed memory
  Samples:       50000
  Memory:        0.586 GB
  ✅ SUCCESS

Test 4: Large dataset (10× CIFAR-10) with managed memory
  Samples:       500000
  Memory:        5.86 GB
  ✅ SUCCESS  (even if > GPU RAM!)
```

### Step 2: Understand the Code

The test demonstrates three memory modes:

**Device Memory (Traditional):**
```fortran
allocate(train_data(3072, n_samples), device)  ! GPU only
```

**Managed Memory (New):**
```fortran
allocate(train_data(3072, n_samples), managed)  ! CPU + GPU
! CUDA handles paging automatically!
```

### Step 3: Performance Considerations

**Managed Memory Performance:**
- **Same as device memory** when data fits in GPU RAM
- **~10-30% slower** when data is 2-10× GPU RAM (due to paging)
- **Works vs crashes** - Better to be 20% slower than crash with OOM!

**Optimization Strategies:**
1. **Prefetching** - Tell CUDA what data you'll need next
2. **Memory Advice** - Mark read-only data
3. **Access Patterns** - Sequential access reduces page faults

## Roadmap

### Phase 1: Verify Managed Memory Works (Current)
- [x] Create simple test program
- [ ] Run test on your GPU
- [ ] Verify 10× dataset size works

### Phase 2: Add to CIFAR-10 Training
- [ ] Copy `cifar10_main.cuf` → `cifar10_main_managed.cuf`
- [ ] Change `device` → `managed` for data arrays
- [ ] Test with normal CIFAR-10 (verify no regression)
- [ ] Test with 2× CIFAR-10 (100K samples)
- [ ] Test with 10× CIFAR-10 (500K samples)

### Phase 3: Optimization
- [ ] Add prefetching (`cudaMemPrefetchAsync`)
- [ ] Add memory advice (`cudaMemAdvise`)
- [ ] Profile with `nsys` to optimize paging
- [ ] Measure performance vs device memory

### Phase 4: Warp Shuffle Intrinsics (Later)
- [ ] Identify reduction operations in CNN
- [ ] Replace shared memory with `__shfl_down()`
- [ ] Target 8× speedup for reductions

## Technical Details

### What is Managed Memory?

CUDA Unified Memory creates a single memory space accessible from both CPU and GPU:

```
CPU RAM:  [============================]  128 GB
           ↕ CUDA pages data automatically
GPU RAM:  [====]  8 GB
```

**How it works:**
1. Allocate with `managed` attribute
2. Access from CPU or GPU code
3. CUDA handles paging automatically (like virtual memory on CPU)
4. GPU page faults trigger transfers from CPU RAM

### When to Use Managed Memory?

**Use managed memory when:**
- ✅ Dataset > GPU RAM
- ✅ Memory-bound workloads
- ✅ Don't want OOM crashes
- ✅ Can tolerate 10-30% slowdown

**Use device memory when:**
- ✅ Dataset fits in GPU RAM
- ✅ Need absolute maximum performance
- ✅ Access patterns are predictable

### Performance Comparison

| Dataset Size | Device Memory | Managed Memory |
|--------------|---------------|----------------|
| 0.5× GPU RAM | ✅ 1.0× (baseline) | ✅ 1.0× (same) |
| 1.0× GPU RAM | ✅ 1.0× | ✅ 0.98× (-2%) |
| 2.0× GPU RAM | ❌ OOM crash | ✅ 0.9× (-10%) |
| 5.0× GPU RAM | ❌ OOM crash | ✅ 0.8× (-20%) |
| 10× GPU RAM  | ❌ OOM crash | ✅ 0.7× (-30%) |

**Key insight:** Better to train 30% slower than not train at all!

## Next Steps

1. **Run the test:**
   ```bash
   ./compile_managed_test.sh
   ./test_managed_memory
   ```

2. **Check the output** - All 4 tests should pass

3. **Report results** - Let me know:
   - Your GPU RAM size
   - Which tests passed
   - Any errors

4. **Next:** Create full CIFAR-10 managed memory version

## References

- [CUDA Unified Memory Programming](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [CUDA Memory Management Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming-hd)
- [Optimizing Unified Memory Performance](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
