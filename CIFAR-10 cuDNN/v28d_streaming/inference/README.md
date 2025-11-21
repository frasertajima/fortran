# Model Export & Inference Guide

This directory contains tools for exporting trained v28 models and loading them for inference.

## 📦 Components

### Fortran Export Module
- **`../common/model_export.cuf`** - Fortran module for exporting trained models
- Saves all weights, biases, and BatchNorm parameters to binary files
- Creates human-readable metadata file

### Python Loader
- **`model_loader.py`** - Python module to load exported models into PyTorch
- Handles Fortran binary format (column-major → row-major conversion)
- Returns ready-to-use PyTorch model in eval mode

### Jupyter Notebook
- **`inference_demo.ipynb`** - Interactive inference demonstrations
- Visualizations, predictions, confusion matrices

---

## 🚀 Quick Start

### Step 1: Export Model from Fortran Training

Add to your training code (e.g., `cifar10_main.cuf`):

```fortran
program train_cifar10
    use dataset_config
    use model_export  ! Add this line
    implicit none

    ! ... your existing training code ...

    ! After training completes, export the model:
    call create_export_directory("saved_models/cifar10/", &
                                 "CIFAR-10", test_accuracy, num_epochs, &
                                 "2025-11-17")

    call export_model_generic("saved_models/cifar10/", &
                             model%conv1_weights, model%conv1_bias, &
                             model%conv2_weights, model%conv2_bias, &
                             model%conv3_weights, model%conv3_bias, &
                             model%fc1_weights, model%fc1_bias, &
                             model%fc2_weights, model%fc2_bias, &
                             model%fc3_weights, model%fc3_bias, &
                             model%bn1_scale, model%bn1_bias, &
                             model%bn1_running_mean, model%bn1_running_var, &
                             model%bn2_scale, model%bn2_bias, &
                             model%bn2_running_mean, model%bn2_running_var, &
                             model%bn3_scale, model%bn3_bias, &
                             model%bn3_running_mean, model%bn3_running_var)

end program train_cifar10
```

### Step 2: Update Compilation Script

Add `model_export.cuf` to your compilation:

```bash
nvfortran -O3 -gpu=cc80 -Mcuda \
  ../common/random_utils.cuf \
  ../common/adam_optimizer.cuf \
  ../common/gpu_batch_extraction.cuf \
  ../common/cuda_utils.cuf \
  ../common/model_export.cuf \    # Add this line
  cifar10_config.cuf \
  cifar10_main.cuf \
  -o cifar10_train \
  -lcudnn -lcublas
```

### Step 3: Load Model in Python

```python
from model_loader import load_v28_model, load_metadata

# Load model
model = load_v28_model('saved_models/cifar10/',
                       in_channels=3,
                       num_classes=10,
                       input_size=32)

# Check metadata
metadata = load_metadata('saved_models/cifar10/')
print(f"Dataset: {metadata['dataset']}")
print(f"Accuracy: {metadata['accuracy']}%")

# Make predictions
import torch
from torchvision import datasets, transforms

# Load test data
transform = transforms.ToTensor()
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)

# Predict on a sample
image, label = test_dataset[0]
image = image.unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1).item()

print(f"True label: {label}")
print(f"Prediction: {prediction}")
```

---

## 📁 Directory Structure After Export

```
saved_models/
└── cifar10/
    ├── model_metadata.txt           # Human-readable info
    ├── conv1_weights.bin            # (32, 3, 3, 3)
    ├── conv1_bias.bin               # (32,)
    ├── conv2_weights.bin            # (64, 32, 3, 3)
    ├── conv2_bias.bin               # (64,)
    ├── conv3_weights.bin            # (128, 64, 3, 3)
    ├── conv3_bias.bin               # (128,)
    ├── fc1_weights.bin              # (512, flatten_size)
    ├── fc1_bias.bin                 # (512,)
    ├── fc2_weights.bin              # (256, 512)
    ├── fc2_bias.bin                 # (256,)
    ├── fc3_weights.bin              # (num_classes, 256)
    ├── fc3_bias.bin                 # (num_classes,)
    ├── bn1_scale.bin                # (32,)
    ├── bn1_bias.bin                 # (32,)
    ├── bn1_running_mean.bin         # (32,)
    ├── bn1_running_var.bin          # (32,)
    ├── bn2_scale.bin                # (64,)
    ├── bn2_bias.bin                 # (64,)
    ├── bn2_running_mean.bin         # (64,)
    ├── bn2_running_var.bin          # (64,)
    ├── bn3_scale.bin                # (128,)
    ├── bn3_bias.bin                 # (128,)
    ├── bn3_running_mean.bin         # (128,)
    └── bn3_running_var.bin          # (128,)
```

---

## 🔍 Understanding the Binary Format

### Fortran Binary Files

Files are written with:
```fortran
open(unit=99, file=filename, form='unformatted', access='stream')
write(99) array
```

This creates:
- **Column-major order** (Fortran default)
- **No headers** (raw binary data)
- **Float32** (4 bytes per value)

### Python Reading

Python reads with:
```python
data = np.fromfile(filepath, dtype=np.float32)
data = data.reshape(shape, order='F')  # Fortran order
```

The key is using `order='F'` to match Fortran's column-major layout.

---

## 📊 Supported Datasets

| Dataset | in_channels | num_classes | input_size | Example |
|---------|-------------|-------------|------------|---------|
| **CIFAR-10** | 3 | 10 | 32 | RGB images |
| **Fashion-MNIST** | 1 | 10 | 28 | Grayscale clothing |
| **CIFAR-100** | 3 | 100 | 32 | Fine-grained RGB |
| **SVHN** | 3 | 10 | 32 | Street view digits |

---

## 🎯 Next Steps

1. ✅ Export module created
2. ✅ Python loader created
3. 🔄 Integrate into training scripts (you do this)
4. 📓 Create Jupyter inference notebook
5. 📊 Add visualization tools

---

## 🐛 Troubleshooting

### "Shape mismatch" error
- Check that `in_channels`, `num_classes`, and `input_size` match your training configuration
- For Fashion-MNIST: `in_channels=1, input_size=28`
- For CIFAR-10/100/SVHN: `in_channels=3, input_size=32`

### "File not found" error
- Ensure model was exported after training
- Check export directory path
- Verify all 31 binary files exist

### Accuracy mismatch
- Ensure model is in `eval()` mode
- Check that BatchNorm running statistics were saved correctly
- Verify test data preprocessing matches training

---

**Created**: 2025-11-17
**Author**: v28 Baseline Team
