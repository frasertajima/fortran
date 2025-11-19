"""
Test different Conv1 weight permutations to find the correct one
that matches Fortran's Conv1 output.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

print("="*70)
print("🔍 Testing Conv1 Weight Permutations")
print("="*70)
print()

# Load Fortran's Conv1 output (golden reference)
debug_dir = Path('datasets/cifar10/saved_models/cifar10')
fortran_conv1 = np.fromfile(debug_dir / 'debug_conv1.bin', dtype=np.float32)
print(f"Fortran Conv1 output size: {len(fortran_conv1)}")

# Reshape from Fortran (W,H,C,1) to Python (1,C,H,W)
fort_whc = fortran_conv1.reshape((32, 32, 32, 1), order='F')
golden_chw = np.zeros((1, 32, 32, 32), dtype=np.float32)
for c in range(32):
    for h in range(32):
        for w in range(32):
            golden_chw[0, c, h, w] = fort_whc[w, h, c, 0]

golden_tensor = torch.from_numpy(golden_chw)
print(f"Golden Conv1 shape: {golden_tensor.shape}")
print(f"Golden stats: min={golden_tensor.min():.4f}, max={golden_tensor.max():.4f}, mean={golden_tensor.mean():.4f}")
print()

# Load input
fortran_input = np.fromfile(debug_dir / 'debug_input.bin', dtype=np.float32)
fort_input_whc = fortran_input.reshape((32, 32, 3, 1), order='F')
input_chw = np.zeros((3, 32, 32), dtype=np.float32)
for c in range(3):
    for h in range(32):
        for w in range(32):
            input_chw[c, h, w] = fort_input_whc[w, h, c, 0]

input_tensor = torch.from_numpy(input_chw).unsqueeze(0)
print(f"Input shape: {input_tensor.shape}")
print()

# Load Conv1 weights
conv1_w_raw = np.fromfile(debug_dir / 'conv1_weights.bin', dtype=np.float32)
conv1_b = np.fromfile(debug_dir / 'conv1_bias.bin', dtype=np.float32)

print(f"Conv1 weights size: {len(conv1_w_raw)} (expected: {32*3*3*3} = {32*3*3*3})")
print(f"Conv1 bias size: {len(conv1_b)}")
print()

# Test different permutations
print("Testing permutations...")
print("="*70)

# Current loading (baseline)
w_baseline = conv1_w_raw.reshape((3, 3, 3, 32), order='F').transpose(3, 2, 0, 1)
w_baseline_t = torch.from_numpy(w_baseline)
b_t = torch.from_numpy(conv1_b)

out_baseline = F.conv2d(input_tensor, w_baseline_t, b_t, padding=1)
corr_baseline = np.corrcoef(golden_tensor.flatten().numpy(), out_baseline.flatten().numpy())[0, 1]
print(f"Baseline (3,3,3,32) F-order -> transpose(3,2,0,1): Corr = {corr_baseline:.4f}")

# Try all 24 permutations of (3,2,0,1)
import itertools
best_corr = corr_baseline
best_perm = (3, 2, 0, 1)
best_output = out_baseline

for perm in itertools.permutations([0, 1, 2, 3]):
    try:
        w_test = conv1_w_raw.reshape((3, 3, 3, 32), order='F').transpose(*perm)
        if w_test.shape != (32, 3, 3, 3):
            continue
        w_test_t = torch.from_numpy(np.ascontiguousarray(w_test))
        out_test = F.conv2d(input_tensor, w_test_t, b_t, padding=1)
        corr = np.corrcoef(golden_tensor.flatten().numpy(), out_test.flatten().numpy())[0, 1]
        
        if corr > 0.99:
            print(f"✅ MATCH! Permutation {perm}: Corr = {corr:.6f}")
            best_corr = corr
            best_perm = perm
            best_output = out_test
        elif corr > best_corr:
            print(f"   Better: {perm}: Corr = {corr:.4f}")
            best_corr = corr
            best_perm = perm
            best_output = out_test
    except:
        pass

print()
print("="*70)
print(f"Best permutation: {best_perm}")
print(f"Best correlation: {best_corr:.6f}")
print()

if best_corr > 0.99:
    print("🎉 FOUND THE CORRECT PERMUTATION!")
    print(f"   Use: reshape((3,3,3,32), order='F').transpose{best_perm}")
else:
    print("⚠️  No perfect match found. Trying different reshape orders...")
    
    # Try C-order reshape
    for perm in itertools.permutations([0, 1, 2, 3]):
        try:
            w_test = conv1_w_raw.reshape((3, 3, 3, 32), order='C').transpose(*perm)
            if w_test.shape != (32, 3, 3, 3):
                continue
            w_test_t = torch.from_numpy(np.ascontiguousarray(w_test))
            out_test = F.conv2d(input_tensor, w_test_t, b_t, padding=1)
            corr = np.corrcoef(golden_tensor.flatten().numpy(), out_test.flatten().numpy())[0, 1]
            
            if corr > 0.99:
                print(f"✅ MATCH with C-order! Permutation {perm}: Corr = {corr:.6f}")
                print(f"   Use: reshape((3,3,3,32), order='C').transpose{perm}")
                break
            elif corr > best_corr:
                print(f"   C-order {perm}: Corr = {corr:.4f}")
        except:
            pass

print("="*70)
