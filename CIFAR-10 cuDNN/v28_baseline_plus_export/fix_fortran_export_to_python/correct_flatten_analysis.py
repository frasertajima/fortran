"""
CORRECT analysis: Fortran uses (W, H, C, N) layout, not (N, C, H, W)!

Fortran: pool3_out(W=4, H=4, C=128, N=batch)
  Flattens to: flatten(N, 2048) where 2048 = 128*4*4
  
  The flatten code does:
    for j in 0..2047:
      k = j / 16  # Channel (0-127)
      spatial_idx = j % 16  # Spatial position (0-15)
      w = spatial_idx % 4
      h = spatial_idx / 4
      flatten[i, j] = pool3[w, h, k, i]
  
  This means: Channel 0 (all 16 positions), Channel 1 (all 16 positions), ...
  Within each channel: positions are in row-major order (h varies slower than w)

PyTorch: pool3_out(N=batch, C=128, H=4, W=4)
  Flattens with .view(N, -1) in C-order: W fastest, then H, then C
  
  This gives: Channel 0 (all 16 positions in row-major), Channel 1, ...
  
So BOTH iterate channels first, then spatial positions in row-major!
They should be the SAME!

Unless... wait, let me check if Fortran's spatial indexing is different.
"""

import numpy as np

C, H, W = 2, 2, 2

print("="*70)
print("Fortran (W, H, C) vs PyTorch (C, H, W) Flatten")
print("="*70)
print()

# Create test data
test_data = np.arange(C * H * W)

# Fortran layout: (W, H, C)
fortran_tensor = test_data.reshape((W, H, C), order='F')  # F-order!
print("Fortran tensor (W, H, C) with F-order:")
print(fortran_tensor)
print()

# Fortran flatten: for each channel, for each h, for each w
fortran_flat = []
for c in range(C):
    for h in range(H):
        for w in range(W):
            fortran_flat.append(fortran_tensor[w, h, c])

print("Fortran flatten (channel-first, then h, then w):")
print(fortran_flat)
print()

# PyTorch layout: (C, H, W)
pytorch_tensor = test_data.reshape((C, H, W), order='C')  # C-order!
print("PyTorch tensor (C, H, W) with C-order:")
print(pytorch_tensor)
print()

# PyTorch flatten: C-order means W fastest
pytorch_flat = pytorch_tensor.flatten(order='C')
print("PyTorch flatten (C-order: w fastest, then h, then c):")
print(list(pytorch_flat))
print()

# Compare
print("Are they the same?", fortran_flat == list(pytorch_flat))
