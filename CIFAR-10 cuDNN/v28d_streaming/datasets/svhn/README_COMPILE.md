# SVHN v28 Baseline - Quick Start

## One-Command Workflow

```bash
# From the SVHN dataset directory
python prepare_svhn.py && bash compile_svhn.sh && ./svhn_train
```

## Step-by-Step

### 1. Prepare Data (One-Time, ~10 seconds)
```bash
python prepare_svhn.py
```
Creates `svhn_data/*.bin` files (~520MB - more training samples!)

### 2. Compile Training Code (~30 seconds)
```bash
bash compile_svhn.sh
```
Compiles using modular v28 framework:
- Common modules from `../../common/`
- Dataset config from `./svhn_config.cuf`
- Main training program from `./svhn_main.cuf`

### 3. Train Model (~80 seconds)
```bash
./svhn_train
```

Expected results:
- **Accuracy**: ~92-93% (street view digits)
- **Time**: ~80 seconds for 15 epochs
- **Performance**: Excellent recognition of real-world digits

## SVHN Dataset

Street View House Numbers - real-world digit recognition:
- **Training**: 73,257 images (more than CIFAR-10!)
- **Test**: 26,032 images
- **Classes**: 10 (digits 0-9)
- **Challenge**: Real-world images with varying angles, lighting

## Modular Structure

```
../../common/
├── random_utils.cuf          # cuRAND wrapper (imported)
├── adam_optimizer.cuf         # NVIDIA Apex FusedAdam (imported)
├── gpu_batch_extraction.cuf   # GPU-only batching (imported)
└── cuda_utils.cuf             # CUDA scheduling (imported)

./
├── svhn_config.cuf            # Dataset parameters & loading  
├── svhn_main.cuf              # cuDNN training + main program
└── compile_svhn.sh            # Compilation script
```

## Performance Notes

SVHN achieves excellent accuracy:
- **Accuracy**: ~92-93%
- **Longer training**: More samples = more time
- **Real-world validation**: Tests model on actual street numbers

This demonstrates the robustness of the v28 baseline framework across different datasets!
