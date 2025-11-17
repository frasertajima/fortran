#!/bin/bash
# ============================================================================
# Compilation script for Oxford Flowers CUDA Fortran implementation
# ============================================================================

echo "======================================================================"
echo "Compiling Oxford Flowers CUDA Fortran Dense Layer Trainer"
echo "======================================================================"
echo ""

# Compiler
NVFORTRAN=nvfortran

# Compilation flags
FLAGS="-cuda -O3 -lcublas -lcudart -lcudnn -lcurand"

# Source file
SOURCE="oxford_flowers_cudnn.cuf"

# Output binary
OUTPUT="oxford_flowers_cudnn"

echo "Compiler: $NVFORTRAN"
echo "Source:   $SOURCE"
echo "Output:   $OUTPUT"
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
    echo "Make sure you have:"
    echo "  1. Extracted features: python extract_mobilenet_features.py"
    echo "  2. Binary files in: ./oxford_features/"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "Compilation failed!"
    echo "======================================================================"
    echo ""
    exit 1
fi
