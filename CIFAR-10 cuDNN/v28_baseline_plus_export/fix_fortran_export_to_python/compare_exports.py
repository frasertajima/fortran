"""
Compare Fortran export vs PyTorch export for the first 20 values.
This will tell us if the export formats are identical.
"""

import numpy as np

print("="*70)
print("Comparing Fortran vs PyTorch Exports")
print("="*70)
print()

# Load Fortran conv1 weights
fortran_data = np.fromfile('datasets/cifar10/saved_models/cifar10/conv1_weights.bin', dtype=np.float32)
print("Fortran export:")
print(f"  Total elements: {len(fortran_data)}")
print(f"  First 20 values:")
for i in range(20):
    print(f"    [{i}]: {fortran_data[i]:.6f}")
print()

# Load PyTorch conv1 weights
pytorch_data = np.fromfile('pytorch_export_test/conv1_weights.bin', dtype=np.float32)
print("PyTorch export:")
print(f"  Total elements: {len(pytorch_data)}")
print(f"  First 20 values:")
for i in range(20):
    print(f"    [{i}]: {pytorch_data[i]:.6f}")
print()

# Compare
print("="*70)
print("Comparison")
print("="*70)
matches = 0
for i in range(min(20, len(fortran_data), len(pytorch_data))):
    if abs(fortran_data[i] - pytorch_data[i]) < 1e-6:
        matches += 1
    else:
        print(f"  Mismatch at [{i}]: Fortran={fortran_data[i]:.6f}, PyTorch={pytorch_data[i]:.6f}")

print()
if matches == 20:
    print("✅ PERFECT MATCH! First 20 values are identical!")
    print("   Fortran and PyTorch export in the same format!")
else:
    print(f"❌ Only {matches}/20 values match")
    print("   Fortran and PyTorch export in DIFFERENT formats!")
