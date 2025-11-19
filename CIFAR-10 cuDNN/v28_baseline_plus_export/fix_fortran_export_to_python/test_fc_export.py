"""
Final debugging: Compare how PyTorch and Fortran export FC weights.
"""

import numpy as np
import torch
import torch.nn as nn

print("="*70)
print("Understanding FC Weight Export")
print("="*70)
print()

# Create a simple PyTorch Linear layer
fc = nn.Linear(4, 2)  # in=4, out=2
fc.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0],   # out=0
                                 [5.0, 6.0, 7.0, 8.0]])  # out=1
print("PyTorch Linear(4, 2) weight:")
print(fc.weight.data)
print(f"Shape: {fc.weight.shape}")  # Should be (2, 4)
print()

# Export using tofile (what our test does)
fc.weight.data.numpy().tofile('test_fc_pytorch.bin')

# Load back with C-order (what our loader does)
loaded_c = np.fromfile('test_fc_pytorch.bin', dtype=np.float32).reshape((2, 4), order='C')
print("Loaded with C-order:")
print(loaded_c)
print()

# Load back with F-order
loaded_f = np.fromfile('test_fc_pytorch.bin', dtype=np.float32).reshape((2, 4), order='F')
print("Loaded with F-order:")
print(loaded_f)
print()

# Check which matches
if np.allclose(loaded_c, fc.weight.data.numpy()):
    print("✅ C-order matches!")
else:
    print("❌ C-order doesn't match")

if np.allclose(loaded_f, fc.weight.data.numpy()):
    print("✅ F-order matches!")
else:
    print("❌ F-order doesn't match")
