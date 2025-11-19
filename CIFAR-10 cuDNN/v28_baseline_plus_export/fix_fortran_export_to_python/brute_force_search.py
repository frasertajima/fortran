"""
BRUTE FORCE: Test all reasonable reshape/transpose combinations to find EXACT match.

Fortran target:
  conv1_out[0,0,0,0:4] = 0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115
  conv1_out[0,15,15,0] = -0.04665595

We'll test different:
1. Reshape dimensions (all permutations of 32,3,3,3)
2. Transpose operations
3. Order ('F' vs 'C')
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from itertools import permutations

# Load test data CORRECTLY
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
first_image_flat = test_data[0, :]
first_image_chw = first_image_flat.reshape((3, 32, 32), order='C')
input_tensor = torch.from_numpy(first_image_chw).unsqueeze(0)

# Load weights
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_data = np.fromfile(model_dir / "conv1_weights.bin", dtype=np.float32)
conv1_b_data = np.fromfile(model_dir / "conv1_bias.bin", dtype=np.float32)

# Target values from Fortran
target_vals = np.array([0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115])
target_15_15 = -0.04665595

print("="*70)
print("BRUTE FORCE SEARCH FOR EXACT MATCH")
print("="*70)
print(f"Target [0,0,0,0:5]: {target_vals}")
print(f"Target [0,0,15,15]: {target_15_15:.6f}")
print()

best_match = None
best_diff = float('inf')
matches = []

# Test different reshape dimensions
reshape_options = [
    ((32, 3, 3, 3), 'F'),
    ((32, 3, 3, 3), 'C'),
    ((3, 3, 3, 32), 'F'),
    ((3, 3, 3, 32), 'C'),
    ((3, 3, 32, 3), 'F'),
    ((3, 3, 32, 3), 'C'),
    ((3, 32, 3, 3), 'F'),
    ((3, 32, 3, 3), 'C'),
]

# Test different transpose operations
transpose_options = [
    None,  # No transpose
    (0, 1, 2, 3),  # Identity
    (0, 1, 3, 2),
    (0, 2, 1, 3),
    (0, 2, 3, 1),
    (0, 3, 1, 2),
    (0, 3, 2, 1),
    (1, 0, 2, 3),
    (1, 0, 3, 2),
    (1, 2, 0, 3),
    (1, 2, 3, 0),
    (1, 3, 0, 2),
    (1, 3, 2, 0),
    (2, 0, 1, 3),
    (2, 0, 3, 1),
    (2, 1, 0, 3),
    (2, 1, 3, 0),
    (2, 3, 0, 1),
    (2, 3, 1, 0),
    (3, 0, 1, 2),
    (3, 0, 2, 1),
    (3, 1, 0, 2),
    (3, 1, 2, 0),
    (3, 2, 0, 1),
    (3, 2, 1, 0),
]

test_count = 0
for reshape_dim, order in reshape_options:
    for transpose_op in transpose_options:
        test_count += 1
        
        try:
            # Reshape
            w = conv1_w_data.reshape(reshape_dim, order=order)
            
            # Transpose if specified
            if transpose_op is not None:
                w = np.transpose(w, transpose_op)
            
            # Check if result is (32, 3, 3, 3) for PyTorch
            if w.shape != (32, 3, 3, 3):
                continue
                
            w = np.ascontiguousarray(w)
            
            # Test with PyTorch
            conv = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
            conv.weight.data = torch.from_numpy(w)
            conv.bias.data = torch.from_numpy(conv1_b_data)
            
            with torch.no_grad():
                out = conv(input_tensor)
            
            # Check all channels and positions for match
            for c in range(32):
                for h in range(32):
                    for w_pos in range(28):  # Check first 28 positions
                        vals = out[0, c, h, w_pos:w_pos+5].numpy()
                        diff = np.max(np.abs(vals - target_vals))
                        
                        if diff < 0.001:  # Very close match
                            val_15_15 = out[0, c, 15, 15].item()
                            diff_15_15 = abs(val_15_15 - target_15_15)
                            
                            total_diff = diff + diff_15_15
                            
                            if total_diff < best_diff:
                                best_diff = total_diff
                                best_match = {
                                    'reshape': reshape_dim,
                                    'order': order,
                                    'transpose': transpose_op,
                                    'channel': c,
                                    'h': h,
                                    'w': w_pos,
                                    'vals': vals,
                                    'val_15_15': val_15_15,
                                    'diff': total_diff
                                }
                            
                            if total_diff < 0.01:
                                matches.append({
                                    'reshape': reshape_dim,
                                    'order': order,
                                    'transpose': transpose_op,
                                    'channel': c,
                                    'h': h,
                                    'w': w_pos,
                                    'vals': vals,
                                    'val_15_15': val_15_15,
                                    'diff': total_diff
                                })
        
        except Exception as e:
            pass

print(f"Tested {test_count} combinations")
print()

if matches:
    print(f"Found {len(matches)} EXCELLENT matches (diff < 0.01):")
    print()
    for i, m in enumerate(matches[:5]):  # Show top 5
        print(f"Match {i+1}:")
        print(f"  Reshape: {m['reshape']}, order='{m['order']}'")
        print(f"  Transpose: {m['transpose']}")
        print(f"  Channel: {m['channel']}, H: {m['h']}, W: {m['w']}")
        print(f"  Values: {m['vals']}")
        print(f"  Val[15,15]: {m['val_15_15']:.6f}")
        print(f"  Total diff: {m['diff']:.6f}")
        print()
elif best_match:
    print(f"Best match found (diff = {best_diff:.6f}):")
    print(f"  Reshape: {best_match['reshape']}, order='{best_match['order']}'")
    print(f"  Transpose: {best_match['transpose']}")
    print(f"  Channel: {best_match['channel']}, H: {best_match['h']}, W: {best_match['w']}")
    print(f"  Values: {best_match['vals']}")
    print(f"  Val[15,15]: {best_match['val_15_15']:.6f}")
else:
    print("No good matches found!")
