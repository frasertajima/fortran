#!/bin/bash
#================================================================
# Compile Warp Shuffle Test - v28c
#================================================================
# Simple test to verify __shfl_down_sync() works

echo "=================================================="
echo "Compiling Warp Shuffle Test (v28c)"
echo "=================================================="

nvfortran -cuda -O3 warp_shuffle.cuf test_warp_shuffle.cuf -o test_warp_shuffle

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Compilation successful!"
    echo ""
    echo "Run with: ./test_warp_shuffle"
    echo "=================================================="
else
    echo ""
    echo "❌ Compilation failed"
    echo ""
    echo "If __shfl_down_sync not found, try:"
    echo "  - Check nvfortran version (needs recent CUDA)"
    echo "  - May need -gpu=ccXX flag for your GPU"
    echo "=================================================="
fi
