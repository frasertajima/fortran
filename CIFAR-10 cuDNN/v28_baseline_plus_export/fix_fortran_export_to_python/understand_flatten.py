"""
Understand the exact flatten mapping between Fortran and PyTorch.

Fortran: pool3_out(W, H, C, N) flattens to flatten(N, C*H*W)
  - Iterates: for each channel, for each spatial position (row-major within channel)
  - flatten[i, j] = pool3[W, H, C, i] where j = C*(H*W) + spatial_idx
  
PyTorch: pool3_out(N, C, H, W) flattens with .view(N, -1)
  - Iterates in C-order: W fastest, then H, then C
  - flatten[i, j] = pool3[i, C, H, W] where j = C*(H*W) + H*W + W

We need to find the permutation that maps Fortran's order to PyTorch's order.
"""

import numpy as np

# Simulate a small 3D tensor: (C=2, H=2, W=2) = 8 elements
C, H, W = 2, 2, 2

print("="*70)
print("Understanding Flatten Mapping")
print("="*70)
print()

# Create a test tensor with sequential values
test_tensor = np.arange(C * H * W).reshape(C, H, W)
print("Test tensor (C, H, W):")
for c in range(C):
    print(f"  Channel {c}:")
    print(f"    {test_tensor[c]}")
print()

# Fortran flatten: For each channel, iterate spatial positions
# Based on the code: flatten_array(i, j) = pool3_array(mod(idx,4) + 1, idx/4 + 1, k+1, i)
# where k = idx / (H*W), spatial_idx = idx % (H*W)
# This means: for each channel k, for each spatial position (w, h)
fortran_flat = []
for c in range(C):
    for h in range(H):
        for w in range(W):
            # Fortran array is (W, H, C, N), so we access as [w, h, c]
            # But our test is (C, H, W), so we access as [c, h, w]
            fortran_flat.append(test_tensor[c, h, w])

print("Fortran flatten order (channel-first, then spatial):")
print(f"  {fortran_flat}")
print()

# PyTorch flatten: C-order on (C, H, W) means W fastest
pytorch_flat = test_tensor.flatten(order='C')  # C-order: last dim varies fastest
print("PyTorch flatten order (C-order: W fastest):")
print(f"  {list(pytorch_flat)}")
print()

# Find the permutation
print("Mapping:")
for i, val in enumerate(fortran_flat):
    pytorch_idx = list(pytorch_flat).index(val)
    print(f"  Fortran[{i}] = {val} → PyTorch[{pytorch_idx}]")
print()

# The permutation array
perm = [list(pytorch_flat).index(val) for val in fortran_flat]
print(f"Permutation array: {perm}")
print()

# Verify
fortran_reordered = [pytorch_flat[i] for i in perm]
print("Verification:")
print(f"  Fortran order:     {fortran_flat}")
print(f"  Reordered PyTorch: {fortran_reordered}")
print(f"  Match: {fortran_flat == fortran_reordered}")
