"""
Test different Conv1 weight reshape dimensions to find the correct one.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import itertools

print("="*70)
print("🔍 Testing Conv1 Weight Reshape Dimensions")
print("="*70)
print()

# Load Fortran's Conv1 output (golden reference)
debug_dir = Path('datasets/cifar10/saved_models/cifar10')
fortran_conv1 = np.fromfile(debug_dir / 'debug_conv1.bin', dtype=np.float32)

# Reshape from Fortran (W,H,C,1) to Python (1,C,H,W)
fort_whc = fortran_conv1.reshape((32, 32, 32, 1), order='F')
golden_chw = np.zeros((1, 32, 32, 32), dtype=np.float32)
for c in range(32):
    for h in range(32):
        for w in range(32):
            golden_chw[0, c, h, w] = fort_whc[w, h, c, 0]

golden_tensor = torch.from_numpy(golden_chw)

# Load input
fortran_input = np.fromfile(debug_dir / 'debug_input.bin', dtype=np.float32)
fort_input_whc = fortran_input.reshape((32, 32, 3, 1), order='F')
input_chw = np.zeros((3, 32, 32), dtype=np.float32)
for c in range(3):
    for h in range(32):
        for w in range(32):
            input_chw[c, h, w] = fort_input_whc[w, h, c, 0]

input_tensor = torch.from_numpy(input_chw).unsqueeze(0)

# Load Conv1 weights
conv1_w_raw = np.fromfile(debug_dir / 'conv1_weights.bin', dtype=np.float32)
conv1_b = np.fromfile(debug_dir / 'conv1_bias.bin', dtype=np.float32)
b_t = torch.from_numpy(conv1_b)

print(f"Testing different reshape dimensions...")
print(f"Total elements: {len(conv1_w_raw)} = 32 filters × 3 channels × 3×3 kernel")
print()

# Try different reshape orders
shapes_to_try = [
    # (shape, order, description)
    ((3, 3, 3, 32), 'F', "Fortran (H,W,C,K)"),
    ((32, 3, 3, 3), 'F', "Fortran (K,C,H,W)"),
    ((3, 3, 32, 3), 'F', "Fortran (H,W,K,C)"),
    ((32, 3, 3, 3), 'C', "C-order (K,C,H,W)"),
    ((3, 32, 3, 3), 'F', "Fortran (C,K,H,W)"),
]

best_corr = -1
best_config = None

for shape, order, desc in shapes_to_try:
    print(f"\nTrying {desc}: reshape{shape}, order='{order}'")
    
    try:
        w_reshaped = conv1_w_raw.reshape(shape, order=order)
        
        # Try all permutations to get to (32, 3, 3, 3)
        for perm in itertools.permutations([0, 1, 2, 3]):
            try:
                w_test = w_reshaped.transpose(*perm)
                if w_test.shape != (32, 3, 3, 3):
                    continue
                    
                w_test_t = torch.from_numpy(np.ascontiguousarray(w_test))
                out_test = F.conv2d(input_tensor, w_test_t, b_t, padding=1)
                corr = np.corrcoef(golden_tensor.flatten().numpy(), out_test.flatten().numpy())[0, 1]
                
                if corr > 0.99:
                    print(f"  ✅ PERFECT MATCH! transpose{perm}: Corr = {corr:.6f}")
                    best_corr = corr
                    best_config = (shape, order, perm, desc)
                    break
                elif corr > best_corr:
                    print(f"  → Better: transpose{perm}: Corr = {corr:.4f}")
                    best_corr = corr
                    best_config = (shape, order, perm, desc)
            except:
                pass
                
        if best_corr > 0.99:
            break
    except Exception as e:
        print(f"  Error: {e}")

print()
print("="*70)
if best_config and best_corr > 0.99:
    shape, order, perm, desc = best_config
    print(f"🎉 FOUND THE SOLUTION!")
    print(f"   Description: {desc}")
    print(f"   Reshape: reshape{shape}, order='{order}'")
    print(f"   Transpose: transpose{perm}")
    print(f"   Correlation: {best_corr:.6f}")
    print()
    print(f"Python code:")
    print(f"   w = np.fromfile(...).reshape{shape}, order='{order}').transpose{perm}")
elif best_config:
    shape, order, perm, desc = best_config
    print(f"Best match found (not perfect):")
    print(f"   {desc}: reshape{shape}, order='{order}', transpose{perm}")
    print(f"   Correlation: {best_corr:.4f}")
else:
    print("No good match found!")

print("="*70)
