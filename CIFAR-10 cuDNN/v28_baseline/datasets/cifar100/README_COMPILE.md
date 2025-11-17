# CIFAR-100 v28 Baseline - Quick Start

## One-Command Workflow

```bash
# From the CIFAR-100 dataset directory
python prepare_cifar100.py && bash compile_cifar100.sh && ./cifar100_train
```

## Step-by-Step

### 1. Prepare Data (One-Time, ~5 seconds)
```bash
python prepare_cifar100.py
```
Creates `cifar100_data/*.bin` files (~360MB)

### 2. Compile Training Code (~30 seconds)
```bash
bash compile_cifar100.sh
```
Compiles using modular v28 framework:
- Common modules from `../../common/`
- Dataset config from `./cifar100_config.cuf`
- Main training program from `./cifar100_main.cuf`

### 3. Train Model (~52 seconds)
```bash
./cifar100_train
```

Expected results:
- **Accuracy**: ~46-50% (100 classes is challenging!)
- **Time**: ~52 seconds for 15 epochs
- **Performance**: Matches PyTorch reference

## CIFAR-100 vs CIFAR-10

The key difference is **100 classes instead of 10**:
- Same 50,000 training images
- Same 10,000 test images
- Same architecture (may need larger model for better accuracy)
- Lower accuracy expected (100-way classification is harder!)

## Modular Structure

```
../../common/
├── random_utils.cuf          # cuRAND wrapper (imported)
├── adam_optimizer.cuf         # NVIDIA Apex FusedAdam (imported)
├── gpu_batch_extraction.cuf   # GPU-only batching (imported)
└── cuda_utils.cuf             # CUDA scheduling (imported)

./
├── cifar100_config.cuf        # Dataset parameters & loading  
├── cifar100_main.cuf          # cuDNN training + main program
└── compile_cifar100.sh        # Compilation script
```

## Performance Notes

CIFAR-100 is significantly harder than CIFAR-10:
- **CIFAR-10**: ~78-79% accuracy (10 classes)
- **CIFAR-100**: ~46-50% accuracy (100 classes)

Both implementations match PyTorch performance, validating code correctness!
