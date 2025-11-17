#!/bin/bash
# ============================================================================
# Compilation script for Oxford Flowers v3 (Optimal Combination)
# ============================================================================

echo "======================================================================"
echo "Compiling Oxford Flowers v3 (Optimal: Shuffling + Weight Decay)"
echo "======================================================================"
echo ""

# Compiler
NVFORTRAN=nvfortran

# Compilation flags
FLAGS="-cuda -O3 -lcublas -lcudart -lcudnn -lcurand"

# Source file
SOURCE="oxford_flowers_cudnn3.cuf"

# Output binary
OUTPUT="oxford_flowers_cudnn3"

echo "Compiler: $NVFORTRAN"
echo "Source:   $SOURCE"
echo "Output:   $OUTPUT"
echo "Config:   Shuffling: YES | Dropout: 0.2 | Weight Decay: 0.01"
echo "Expected: v2a (+0.39%) + v2c (+0.65%) = Potential best performance"
echo "Flags:    $FLAGS"
echo ""

# Compile
echo "Compiling..."
$NVFORTRAN $FLAGS $SOURCE -o $OUTPUT

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Compilation successful!"
    echo "======================================================================"
    echo ""
    echo "To run:"
    echo "  ./$OUTPUT"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "Compilation failed!"
    echo "======================================================================"
    echo ""
    exit 1
fi
