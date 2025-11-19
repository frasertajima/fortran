"""
COMPREHENSIVE SEARCH: Check ALL channels, ALL positions for the target values.

Maybe Fortran's channel indexing doesn't match what we think.
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

# Load weights - Strategy 2
model_dir = Path("datasets/cifar10/saved_models/cifar10")
conv1_w_data = np.fromfile(model_dir / "conv1_weights.bin", dtype=np.float32)
conv1_b_data = np.fromfile(model_dir / "conv1_bias.bin", dtype=np.float32)

w = conv1_w_data.reshape((3, 3, 3, 32), order='F')
w = np.transpose(w, (3, 2, 0, 1))
w = np.ascontiguousarray(w)

conv = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
conv.weight.data = torch.from_numpy(w)
conv.bias.data = torch.from_numpy(conv1_b_data)

with torch.no_grad():
    out = conv(input_tensor)  # (1, 32, 32, 32)

# Target values
target = np.array([0.1936478, 0.3838937, 0.3868373, 0.3895212, 0.4106115])

print("Searching ALL channels and positions for target pattern...")
print(f"Target: {target}")
print()

best_matches = []

# Search all channels
for c in range(32):
    # Search all H positions
    for h in range(32):
        # Search all W starting positions
        for w_start in range(28):  # Leave room for 5 values
            vals = out[0, c, h, w_start:w_start+5].numpy()
            diff = np.max(np.abs(vals - target))
            
            if diff < 0.1:  # Close match
                best_matches.append({
                    'c': c,
                    'h': h,
                    'w': w_start,
                    'vals': vals,
                    'diff': diff
                })

# Sort by difference
best_matches.sort(key=lambda x: x['diff'])

print(f"Found {len(best_matches)} matches with diff < 0.1")
print()

if best_matches:
    print("Top 10 matches:")
    for i, m in enumerate(best_matches[:10]):
        print(f"{i+1}. Channel {m['c']}, H={m['h']}, W={m['w']}")
        print(f"   Values: {m['vals']}")
        print(f"   Diff: {m['diff']:.6f}")
        print()
else:
    print("No close matches found!")
    print("\nLet's check the closest match overall:")
    
    min_diff = float('inf')
    best = None
    
    for c in range(32):
        for h in range(32):
            for w_start in range(28):
                vals = out[0, c, h, w_start:w_start+5].numpy()
                diff = np.max(np.abs(vals - target))
                
                if diff < min_diff:
                    min_diff = diff
                    best = {'c': c, 'h': h, 'w': w_start, 'vals': vals, 'diff': diff}
    
    if best:
        print(f"Closest match: Channel {best['c']}, H={best['h']}, W={best['w']}")
        print(f"Values: {best['vals']}")
        print(f"Diff: {best['diff']:.6f}")
