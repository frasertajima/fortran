# CIFAR-10 v28 Baseline - Quick Start

## One-Command Workflow

```bash
# From the CIFAR-10 dataset directory
python prepare_cifar10.py && bash compile_cifar10.sh && ./cifar10_train
```

## Step-by-Step

### 1. Prepare Data (One-Time, ~5 seconds)
```bash
python prepare_cifar10.py
```
Creates `cifar10_data/*.bin` files (~360MB)

### 2. Compile Training Code (~30 seconds)
```bash
bash compile_cifar10.sh
```
Compiles using modular v28 framework:
- Common modules from `../../common/`
- Dataset config from `./cifar10_config.cuf`
- Main training program from `./cifar10_main.cuf`

### 3. Train Model (~31 seconds)
```bash
./cifar10_train
```

Expected results:
- **Accuracy**: ~78-79% (matches PyTorch reference)
- **Time**: ~31 seconds for 15 epochs  
- **Speed**: 2x faster than PyTorch!

## Modular Structure

This training uses the v28 baseline modular architecture:

```
../../common/
├── random_utils.cuf          # cuRAND wrapper (imported)
├── adam_optimizer.cuf         # NVIDIA Apex FusedAdam (imported)
├── gpu_batch_extraction.cuf   # GPU-only batching (imported)
└── cuda_utils.cuf             # CUDA scheduling (imported)

./
├── cifar10_config.cuf         # Dataset parameters & loading  
├── cifar10_main.cuf           # cuDNN training + main program
└── compile_cifar10.sh         # Compilation script
```

## Note on Main Training File

The current `cifar10_main.cuf` includes the full cuDNN training module inline. 
This will be refactored to use common modules in a future update.

For now, the main code reuse benefits come from:
- ✅ Optimizer (100% shared)
- ✅ Batch extraction (100% shared) 
- ✅ Random utilities (100% shared)
- ✅ CUDA utilities (100% shared)

Next step: Extract cuDNN layers to common modules for even more reuse!
