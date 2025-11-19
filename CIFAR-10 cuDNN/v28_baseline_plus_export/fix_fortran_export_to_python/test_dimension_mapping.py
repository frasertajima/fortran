"""
KEY INSIGHT: Fortran array is (W,H,C,N) but cuDNN writes in NCHW order!

When cuDNN writes to conv1_out_gpu with NCHW descriptor (1, 32, 32, 32),
it writes: N=0, C=0-31, H=0-31, W=0-31

But Fortran allocated as (W, H, C, N) = (32, 32, 32, 1)

So when Fortran reads conv1_out_host(1,1,1,1), it's reading:
- Fortran index (W=1, H=1, C=1, N=1)
- But cuDNN wrote it as (N=?, C=?, H=?, W=?)

We need to figure out the mapping!
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load test data
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
first_image_flat = test_data[0, :]
first_image_chw = first_image_flat.reshape((3, 32, 32), order='C')
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)

# Load weights - try Strategy 2 which was closest
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_data = np.fromfile(model_dir / "conv1_weights.bin", dtype=np.float32)
conv1_b_data = np.fromfile(model_dir / "conv1_bias.bin", dtype=np.float32)

# Strategy 2
w = conv1_w_data.reshape((3, 3, 3, 32), order='F')
w = np.transpose(w, (3, 2, 0, 1))
w = np.ascontiguousarray(w)

conv = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
conv.weight.data = torch.from_numpy(w)
conv.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    out = conv(input_tensor)  # Shape: (1, 32, 32, 32) = (N, C, H, W)

print("PyTorch output shape (N, C, H, W):", out.shape)
print()

# Fortran reads (W, H, C, N) but cuDNN wrote (N, C, H, W)
# So Fortran (W=1, H=1, C=1, N=1) might map to PyTorch (N=?, C=?, H=?, W=?)

# Let's check all possible interpretations:
print("Checking different dimension mappings:")
print()

# Hypothesis 1: Direct mapping (unlikely)
print("Hypothesis 1: Fortran (W,H,C,N) = PyTorch (W,H,C,N)")
print(f"  Fortran (1,1,1,1) = PyTorch [0,0,0,0]: {out[0,0,0,0].item():.6f}")
print(f"  Target: 0.1936478")
print()

# Hypothesis 2: Fortran W maps to PyTorch C
print("Hypothesis 2: Fortran (W,H,C,N) = PyTorch (N,W,H,C)")
print(f"  Fortran (1-5,1,1,1) = PyTorch [0,0:5,0,0]: {out[0,0:5,0,0].numpy()}")
print(f"  Target: [0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115]")
print()

# Hypothesis 3: Fortran reads transposed
print("Hypothesis 3: Fortran (W,H,C,N) = PyTorch (N,C,W,H)")
print(f"  Fortran (1-5,1,1,1) = PyTorch [0,0,0:5,0]: {out[0,0,0:5,0].numpy()}")
print(f"  Target: [0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115]")
print()

# Hypothesis 4: Complete permutation
print("Hypothesis 4: Fortran (W,H,C,N) = PyTorch (N,H,W,C)")
print(f"  Fortran (1-5,1,1,1) = PyTorch [0,0,0,0:5]: {out[0,0,0,0:5].numpy()}")
print(f"  Target: [0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115]")
