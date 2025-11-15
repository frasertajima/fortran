#!/bin/bash
# ============================================================================
# Compilation script for Oxford Flowers v2b (Dropout 0.3 Only)
# ============================================================================

echo "======================================================================"
echo "Compiling Oxford Flowers v2b (Dropout 0.3 Only)"
echo "======================================================================"
echo ""

# Compiler
NVFORTRAN=nvfortran

# Compilation flags
FLAGS="-cuda -O3 -lcublas -lcudart -lcudnn -lcurand"

# Source file
SOURCE="oxford_flowers_cudnn2b.cuf"

# Output binary
OUTPUT="oxford_flowers_cudnn2b"

echo "Compiler: $NVFORTRAN"
echo "Source:   $SOURCE"
echo "Output:   $OUTPUT"
echo "Config:   Shuffling: NO | Dropout: 0.3 | Weight Decay: 0.0"
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
