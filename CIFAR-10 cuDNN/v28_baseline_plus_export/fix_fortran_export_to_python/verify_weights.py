"""
Compare Fortran text export with Python loaded values.
This will systematically verify each step.
"""

import numpy as np

print("="*70)
print("Systematic Weight Verification")
print("="*70)
print()

# Read Fortran text export
print("Step 1: Reading Fortran text export...")
fortran_values = {}
with open('datasets/cifar10/saved_models/cifar10/conv1_weights.bin.txt', 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 5:
            k, c, h, w = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            value = float(parts[4])
            fortran_values[(k, c, h, w)] = value

print(f"  Found {len(fortran_values)} values in Fortran export")
print(f"  First few values:")
for key in sorted(fortran_values.keys())[:5]:
    print(f"    [{key[0]},{key[1]},{key[2]},{key[3]}] = {fortran_values[key]:.6f}")
print()

# Load binary with Python
print("Step 2: Loading binary with Python (F-order reshape)...")
conv1_data = np.fromfile('datasets/cifar10/saved_models/cifar10/conv1_weights.bin', dtype=np.float32)
conv1_weights = conv1_data.reshape((32, 3, 3, 3), order='F')
print(f"  Loaded shape: {conv1_weights.shape}")
print(f"  First few values from Python:")
for k, c, h, w in sorted(fortran_values.keys())[:5]:
    # Convert 1-indexed to 0-indexed
    py_value = conv1_weights[k-1, c-1, h-1, w-1]
    print(f"    [{k},{c},{h},{w}] = {py_value:.6f}")
print()

# Compare
print("Step 3: Comparing Fortran export vs Python load...")
matches = 0
mismatches = 0
for (k, c, h, w), fortran_val in fortran_values.items():
    py_val = conv1_weights[k-1, c-1, h-1, w-1]  # Convert to 0-indexed
    if abs(fortran_val - py_val) < 1e-6:
        matches += 1
    else:
        mismatches += 1
        if mismatches <= 5:  # Show first 5 mismatches
            print(f"  ❌ Mismatch at [{k},{c},{h},{w}]:")
            print(f"     Fortran: {fortran_val:.6f}")
            print(f"     Python:  {py_val:.6f}")

print()
if mismatches == 0:
    print(f"✅ PERFECT MATCH! All {matches} values match!")
    print("The F-order loading is working correctly!")
else:
    print(f"❌ Found {mismatches} mismatches out of {matches + mismatches} values")
    print("The F-order loading is NOT working correctly")

print()
print("="*70)
print("Additional Analysis")
print("="*70)

# Try C-order loading for comparison
print("\nTrying C-order reshape for comparison...")
conv1_weights_c = conv1_data.reshape((32, 3, 3, 3), order='C')
matches_c = 0
for (k, c, h, w), fortran_val in fortran_values.items():
    py_val = conv1_weights_c[k-1, c-1, h-1, w-1]
    if abs(fortran_val - py_val) < 1e-6:
        matches_c += 1

print(f"  F-order matches: {matches}/{len(fortran_values)}")
print(f"  C-order matches: {matches_c}/{len(fortran_values)}")
