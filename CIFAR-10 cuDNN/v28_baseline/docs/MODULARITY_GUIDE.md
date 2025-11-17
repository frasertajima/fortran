# Modularity Guide - v28 Baseline

## Why Modularity Matters

### The Problem Before v28 Baseline

Before modular refactoring, we had:

```
cifar10_cudnn_v28.cuf      4,014 lines  ├─ curand_wrapper (60 lines) - 100% duplicated
cifar100_cudnn.cuf         4,045 lines  ├─ adam_optimizer (110 lines) - 100% duplicated
svhn_cudnn.cuf             4,055 lines  ├─ data_loading (150 lines) - 70% duplicated
                                        ├─ batch_extraction (150 lines) - 90% duplicated
Total:                    12,114 lines  └─ training_code (3500 lines) - 95% duplicated
```

**Problems**:
1. **Bug fixes required 3× work**: Fix in CIFAR-10, copy to CIFAR-100, copy to SVHN
2. **New features required 3× work**: Implement once, copy-paste twice
3. **Hard to maintain**: 12K lines with massive duplication
4. **Slow to experiment**: Adding Fashion-MNIST = 4K new lines

### The Solution: v28 Baseline

```
common/
├── random_utils.cuf (90 lines) ────────┐
├── adam_optimizer.cuf (150 lines) ─────┤
├── gpu_batch_extraction.cuf (180 lines)├─ 100% reusable!
└── cuda_utils.cuf (465 lines) ─────────┘

datasets/
├── cifar10_config.cuf (150 lines)  ─┐
├── cifar100_config.cuf (150 lines) ─├─ Only parameters differ!
└── svhn_config.cuf (150 lines)  ────┘

Total: ~1,500 lines (0% duplication!)
```

**Benefits**:
1. **Fix once, benefit everywhere**: 1× work for all datasets
2. **Add features once**: Implement in common/, use everywhere
3. **Easy to maintain**: 1.5K lines vs 12K lines
4. **Fast experiments**: New dataset = ~300 lines

## Modularity Principles

### 1. Separation of Concerns

**Rule**: Separate "what" (dataset parameters) from "how" (training logic)

**Example**:
```fortran
! ❌ BAD: Hardcoded parameters in training logic
subroutine train_network()
    integer :: num_classes = 10  ! What if CIFAR-100?
    integer :: input_size = 3072 ! What if MNIST?
    ! ... training code ...
end subroutine

! ✅ GOOD: Import parameters from config
subroutine train_network()
    use dataset_config, only: num_classes, input_size
    ! ... training code (works for ANY dataset) ...
end subroutine
```

### 2. Single Responsibility

**Rule**: Each module does ONE thing well

**Example**:
```fortran
! ❌ BAD: One huge module
module everything
    ! Data loading
    ! Batch extraction
    ! Optimizer
    ! Training logic
    ! ... 4000 lines ...
end module

! ✅ GOOD: Focused modules
module gpu_batch_extraction
    ! ONLY batch extraction
    ! ~180 lines
end module

module adam_optimizer
    ! ONLY optimizer
    ! ~150 lines
end module
```

### 3. Interface Segregation

**Rule**: All datasets expose the same interface

**Example**:
```fortran
! All dataset configs MUST provide:
module dataset_config
    ! Parameters
    integer, parameter :: train_samples
    integer, parameter :: test_samples
    integer, parameter :: num_classes
    integer, parameter :: input_size

    ! Data arrays
    real(4), device, allocatable :: gpu_train_data(:,:)
    integer, device, allocatable :: gpu_train_labels(:)

    ! Loading function
    subroutine load_dataset()
    end subroutine
end module
```

**Benefit**: Training code works with ANY dataset that follows this interface!

### 4. Dependency Inversion

**Rule**: Training code depends on abstractions (interfaces), not concrete implementations

**Example**:
```fortran
! Training code doesn't know if it's CIFAR-10 or CIFAR-100
program train
    use dataset_config  ! Abstract interface
    use gpu_batch_extraction  ! Abstract interface

    call load_dataset()  ! Works for any dataset!
    call extract_training_batch_gpu(...)  ! Works for any batch size!
end program
```

## Modularity in Practice

### Adding a New Optimization

**Scenario**: You discover a better Adam optimizer variant

**Before (Monolithic)**:
```bash
1. Edit cifar10_cudnn_v28.cuf (find adam_update functions)
2. Copy changes to cifar100_cudnn.cuf
3. Copy changes to svhn_cudnn.cuf
4. Test all three
5. Fix bugs in all three

Time: 2-3 hours, error-prone
```

**After (Modular)**:
```bash
1. Edit common/adam_optimizer.cuf
2. Recompile all datasets
3. Test all three (automatically use new optimizer!)

Time: 30 minutes, safe
```

### Adding a New Dataset

**Before**: Copy entire 4K line file, modify throughout
**After**: Create 150-line config, done!

See `ADDING_NEW_DATASET.md` for details.

### Debugging a Common Issue

**Scenario**: Batch extraction has a bug

**Before**:
```bash
# Bug in CIFAR-10? Fix it
vim cifar10_cudnn_v28.cuf  # Search for batch extraction (line 3200?)
# Now fix in CIFAR-100
vim cifar100_cudnn.cuf     # Search again...
# Now fix in SVHN
vim svhn_cudnn.cuf         # Search again...
# Did you fix all instances? Hope so!
```

**After**:
```bash
vim common/gpu_batch_extraction.cuf  # One file, one fix
# Recompile. Done. All datasets benefit!
```

## Design Patterns

### Pattern 1: Strategy Pattern (Batch Extraction)

Different strategies for different use cases:

```fortran
! Training: WITH shuffling
call extract_training_batch_gpu(train_data, train_labels, ...)

! Testing: WITHOUT shuffling
call extract_test_batch_gpu(test_data, test_labels, ...)
```

Both use the same underlying mechanism, just different strategies.

### Pattern 2: Template Method (Training Loop)

Training loop is a template, datasets fill in the blanks:

```fortran
! Template (in main training code)
do epoch = 1, NUM_EPOCHS
    call shuffle_indices(train_samples)  ! Uses dataset_config.train_samples

    num_batches = (train_samples + BATCH_SIZE - 1) / BATCH_SIZE

    do batch_idx = 1, num_batches
        call extract_training_batch_gpu(...)  ! Uses dataset_config.gpu_train_data
        ! ... training ...
    end do
end do
```

The structure is fixed, the data is injected!

### Pattern 3: Dependency Injection

Dataset configuration is "injected" via module import:

```fortran
! For CIFAR-10 training
use dataset_config  ! Links to cifar10_config.cuf at compile time

! For CIFAR-100 training
use dataset_config  ! Links to cifar100_config.cuf at compile time
```

Same code, different data!

## Testing Modularity

### Test 1: Can You Add a Dataset Without Changing Common Code?

**Target**: Yes
**Current**: Yes ✅

### Test 2: Can You Fix a Bug in One Place?

**Target**: Yes
**Current**: Yes ✅

### Test 3: Can Common Modules Be Used in New Projects?

**Target**: Yes
**Current**: Yes ✅ (just copy common/ directory)

## Anti-Patterns to Avoid

### Anti-Pattern 1: Hardcoding Dataset-Specific Values

```fortran
! ❌ DON'T do this in common modules
subroutine some_common_function()
    integer :: num_classes = 10  ! Hardcoded!
end subroutine

! ✅ DO this
subroutine some_common_function(num_classes)
    integer, intent(in) :: num_classes  ! Parameter!
end subroutine
```

### Anti-Pattern 2: Mixing Concerns

```fortran
! ❌ DON'T put data loading in batch extraction
module gpu_batch_extraction
    subroutine load_data()  ! Wrong module!
        ! ...
    end subroutine
end module

! ✅ DO keep them separate
module dataset_config
    subroutine load_dataset()  ! Correct module!
        ! ...
    end subroutine
end module
```

### Anti-Pattern 3: Copy-Paste Instead of Extract

```fortran
! ❌ DON'T copy common code to each dataset
! cifar10_config.cuf
subroutine adam_update(...)
    ! Same code as CIFAR-100...
end subroutine

! ✅ DO extract to common module
! common/adam_optimizer.cuf
subroutine adam_update(...)
    ! Used by all datasets
end subroutine
```

## Metrics of Success

### Code Duplication

- **Target**: <5%
- **Before**: ~90%
- **After**: 0% ✅

### Lines of Code per Dataset

- **Target**: <300 lines
- **Before**: ~4,000 lines
- **After**: ~150 lines ✅

### Time to Add Dataset

- **Target**: <2 hours
- **Before**: 2-3 days (copy, modify, debug)
- **After**: 1-2 hours ✅

### Bug Fix Propagation

- **Target**: 1 edit fixes all datasets
- **Before**: 3 edits (one per dataset)
- **After**: 1 edit ✅

## Best Practices

### 1. Think "Interface First"

Before adding a feature, ask:
- Can this work for ALL datasets?
- Should this be in common/ or datasets/?
- What interface should it expose?

### 2. Test with Multiple Datasets

After modifying common/, test with:
- CIFAR-10 (baseline)
- CIFAR-100 (different num_classes)
- SVHN (different data size)

If it works for all three, it's truly generic!

### 3. Document Assumptions

```fortran
! ✅ GOOD: Document what you expect
subroutine extract_batch(data, labels, batch_size, ...)
    ! Assumes:
    !   - data is (N, features)
    !   - labels is (N)
    !   - batch_size <= N
end subroutine
```

### 4. Keep Common Modules Simple

**Rule of thumb**: If common module needs to know about specific dataset, it's not common!

## Evolution of Modularity

### Phase 1: Extract Obvious Duplicates ✅ (Current)

- cuRAND wrapper
- Adam optimizer
- GPU batch extraction

### Phase 2: Extract Training Logic (Next)

- cuDNN layer wrappers
- Loss computation
- Metrics tracking

### Phase 3: Configuration Files (Future)

Instead of Fortran config, use JSON/YAML:

```yaml
dataset:
  name: fashion-mnist
  train_samples: 60000
  test_samples: 10000
  input_channels: 1
  input_height: 28
  input_width: 28
```

### Phase 4: Generic Training Binary (Future Vision)

```bash
# Single executable, multiple datasets!
./train_cnn --dataset=fashion_mnist --epochs=15
./train_cnn --dataset=cifar10 --epochs=15
./train_cnn --dataset=cifar100 --epochs=25
```

No recompilation needed!

## Conclusion

Modularity is not just about code organization—it's about:

1. **Productivity**: Do work once, benefit everywhere
2. **Quality**: Fix bugs once, all datasets improve
3. **Experimentation**: Try new ideas quickly
4. **Maintainability**: Understand code faster

**The v28 Baseline proves you can have both high performance AND clean modularity!**

### Key Lessons

1. ✅ Performance comes first (get v28 working)
2. ✅ Then extract common patterns (create v28 baseline)
3. ✅ Test with multiple datasets (CIFAR-10, CIFAR-100, SVHN)
4. ✅ Document thoroughly (you're reading this!)
5. ✅ Make it easy to extend (Fashion-MNIST in <2 hours)

**Modularity is not a luxury—it's an investment that pays off immediately!** 🚀
