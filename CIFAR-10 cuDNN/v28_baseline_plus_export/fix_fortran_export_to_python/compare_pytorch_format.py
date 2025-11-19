"""
Compare PyTorch's native weight format with our loaded weights.
This will show us if the issue is in how we're interpreting the binary data.
"""

import numpy as np
import torch

print("="*70)
print("Comparing PyTorch Native vs Our Loaded Weights")
print("="*70)
print()

# Load the PyTorch-exported conv1 weights
pytorch_conv1_data = np.fromfile('pytorch_export_test/conv1_weights.bin', dtype=np.float32)
print(f"PyTorch conv1 binary: {len(pytorch_conv1_data)} elements")
print(f"Expected: 32 * 3 * 3 * 3 = {32*3*3*3}")
print(f"First 10 values: {pytorch_conv1_data[:10]}")
print()

# PyTorch stores weights as (out_channels, in_channels, H, W) in C-order
# So for conv1: (32, 3, 3, 3) in C-order
pytorch_shape = (32, 3, 3, 3)

# Try loading with F-order (what our loader does)
loaded_f = pytorch_conv1_data.reshape(pytorch_shape, order='F')
print("Loaded with F-order (what our loader does):")
print(f"  Shape: {loaded_f.shape}")
print(f"  [0,0,0,0] = {loaded_f[0,0,0,0]:.6f}")
print(f"  [0,0,0,1] = {loaded_f[0,0,0,1]:.6f}")
print(f"  [0,0,1,0] = {loaded_f[0,0,1,0]:.6f}")
print()

# Try loading with C-order (PyTorch's native format)
loaded_c = pytorch_conv1_data.reshape(pytorch_shape, order='C')
print("Loaded with C-order (PyTorch's native format):")
print(f"  Shape: {loaded_c.shape}")
print(f"  [0,0,0,0] = {loaded_c[0,0,0,0]:.6f}")
print(f"  [0,0,0,1] = {loaded_c[0,0,0,1]:.6f}")
print(f"  [0,0,1,0] = {loaded_c[0,0,1,0]:.6f}")
print()

print("="*70)
print("INSIGHT")
print("="*70)
print()
print("PyTorch stores weights in C-order (row-major)!")
print("But we're loading with F-order!")
print()
print("This is why the PyTorch export/load cycle fails!")
print("We need to load PyTorch-exported weights with C-order,")
print("but Fortran-exported weights with F-order.")
