#!/bin/bash
# Compile managed memory test

set -e

echo "Compiling managed memory test..."

nvfortran -cuda -O3 test_managed_memory.cuf -o test_managed_memory

echo "✅ Compilation successful!"
echo ""
echo "Run the test:"
echo "  ./test_managed_memory"
echo ""
echo "Expected output:"
echo "  - Test 1-3 should succeed (fit in GPU RAM)"
echo "  - Test 4 should succeed with managed memory (even if > GPU RAM)"
