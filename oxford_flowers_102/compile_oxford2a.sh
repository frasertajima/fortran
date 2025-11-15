#!/bin/bash
# ============================================================================
# Compilation script for Oxford Flowers v2a (Shuffling Only)
# ============================================================================

echo "======================================================================"
echo "Compiling Oxford Flowers v2a (Shuffling Only)"
echo "======================================================================"
echo ""

# Compiler
NVFORTRAN=nvfortran

# Compilation flags
FLAGS="-cuda -O3 -lcublas -lcudart -lcudnn -lcurand"

# Source file
SOURCE="oxford_flowers_cudnn2a.cuf"

# Output binary
OUTPUT="oxford_flowers_cudnn2a"

echo "Compiler: $NVFORTRAN"
echo "Source:   $SOURCE"
echo "Output:   $OUTPUT"
echo "Config:   Shuffling: YES | Dropout: 0.2 | Weight Decay: 0.0"
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
