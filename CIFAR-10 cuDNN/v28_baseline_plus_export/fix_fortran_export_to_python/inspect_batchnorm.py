"""
Test Suspect #2: BatchNorm Variance vs Inverse-Std

Check if Fortran exported invStd instead of variance.
"""

import numpy as np

print("="*70)
print("Inspecting BatchNorm Running Statistics")
print("="*70)
print()

for bn_name in ['bn1', 'bn2', 'bn3']:
    print(f"{bn_name.upper()}:")
    
    var = np.fromfile(f'datasets/cifar10/saved_models/cifar10/{bn_name}_running_var.bin', dtype=np.float32)
    mean = np.fromfile(f'datasets/cifar10/saved_models/cifar10/{bn_name}_running_mean.bin', dtype=np.float32)
    scale = np.fromfile(f'datasets/cifar10/saved_models/cifar10/{bn_name}_scale.bin', dtype=np.float32)
    bias = np.fromfile(f'datasets/cifar10/saved_models/cifar10/{bn_name}_bias.bin', dtype=np.float32)
    
    print(f"  Running Variance:")
    print(f"    Min: {var.min():.6f}, Max: {var.max():.6f}, Mean: {var.mean():.6f}")
    print(f"    First 5: {var[:5]}")
    
    print(f"  Running Mean:")
    print(f"    Min: {mean.min():.6f}, Max: {mean.max():.6f}, Mean: {mean.mean():.6f}")
    
    print(f"  Scale (gamma):")
    print(f"    Min: {scale.min():.6f}, Max: {scale.max():.6f}, Mean: {scale.mean():.6f}")
    
    print(f"  Bias (beta):")
    print(f"    Min: {bias.min():.6f}, Max: {bias.max():.6f}, Mean: {bias.mean():.6f}")
    
    print()
    
    # Check if values look like invStd
    if var.min() < 0.001:
        print(f"  ⚠️  WARNING: {bn_name} variance has very small values - might be invStd!")
    elif var.max() > 1000:
        print(f"  ⚠️  WARNING: {bn_name} variance has very large values - might be invVar!")
    else:
        print(f"  ✓ {bn_name} variance looks normal (typical range for variance)")
    print()

print("="*70)
print("Analysis:")
print("  - Normal variance should be in range ~0.01 to ~10")
print("  - If values are < 0.01, might be invStd")
print("  - If values are > 100, might be invVar")
print("="*70)
