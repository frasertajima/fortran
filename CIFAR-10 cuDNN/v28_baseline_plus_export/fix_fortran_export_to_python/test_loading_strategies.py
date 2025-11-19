"""
Test all possible loading strategies for the Fortran-exported array.
Goal: Find which strategy correctly reconstructs the original array.
"""

import numpy as np

print("="*70)
print("Testing Fortran Binary Loading Strategies")
print("="*70)
print()

# Load the text file to see the ground truth
print("Ground truth from text file:")
with open('test_array.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[:10]:  # Show first 10 lines
        print(line.rstrip())
print()

# Expected values based on Fortran output
# test_array(1,1,1,1) = 1.0, test_array(1,1,1,2) = 2.0, etc.
expected = {
    (1,1,1,1): 1.0,
    (1,1,1,2): 2.0,
    (1,1,2,1): 3.0,
    (1,1,2,2): 4.0,
    (1,2,1,1): 5.0,
    (2,1,1,1): 9.0,
    (2,2,2,2): 16.0,
}

print("Expected values (1-indexed, Fortran style):")
for key, val in expected.items():
    print(f"  test_array{key} = {val}")
print()

# Test strategies
strategies = [
    {
        "name": "Strategy 1: F-order reshape (2,2,2,2)",
        "file": "test_array_f_order.bin",
        "load": lambda data: data.reshape((2,2,2,2), order='F')
    },
    {
        "name": "Strategy 2: C-order reshape (2,2,2,2)",
        "file": "test_array_f_order.bin",
        "load": lambda data: data.reshape((2,2,2,2), order='C')
    },
    {
        "name": "Strategy 3: F-order reshape (2,2,2,2) reversed",
        "file": "test_array_f_order.bin",
        "load": lambda data: data.reshape((2,2,2,2), order='F')[::-1,::-1,::-1,::-1]
    },
    {
        "name": "Strategy 4: Load as (W,H,C,K) F-order, transpose to (K,C,H,W)",
        "file": "test_array_f_order.bin",
        "load": lambda data: np.transpose(data.reshape((2,2,2,2), order='F'), (3,2,1,0))
    },
    {
        "name": "Strategy 5: C-order file, C-order reshape",
        "file": "test_array_c_order.bin",
        "load": lambda data: data.reshape((2,2,2,2), order='C')
    },
    {
        "name": "Strategy 6: C-order file, F-order reshape",
        "file": "test_array_c_order.bin",
        "load": lambda data: data.reshape((2,2,2,2), order='F')
    },
]

print("="*70)
print("Testing Strategies")
print("="*70)
print()

for strategy in strategies:
    print(f"{strategy['name']}")
    print("-" * 70)
    
    try:
        # Load binary
        data = np.fromfile(strategy['file'], dtype=np.float32)
        
        # Apply loading strategy
        array = strategy['load'](data)
        
        # Check against expected values (convert to 0-indexed)
        matches = 0
        total = len(expected)
        
        for (k, c, h, w), expected_val in expected.items():
            # Convert 1-indexed to 0-indexed
            actual_val = array[k-1, c-1, h-1, w-1]
            if abs(actual_val - expected_val) < 0.01:
                matches += 1
            else:
                print(f"  ❌ [{k},{c},{h},{w}]: expected {expected_val}, got {actual_val}")
        
        if matches == total:
            print(f"  ✅ PERFECT MATCH! All {total} values correct!")
            print(f"  This is the correct loading strategy!")
            print()
            print("  Full array:")
            print(array)
        else:
            print(f"  ⚠️  Partial match: {matches}/{total} correct")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()

print("="*70)
print("Summary")
print("="*70)
print("The strategy that shows '✅ PERFECT MATCH' is the correct one!")
