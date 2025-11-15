#!/bin/bash
# =======================================================================
# Compilation script for CIFAR-10 v27 (Python Preprocessing)
# =======================================================================

echo "======================================================================="
echo "Compiling CIFAR-10 v27 (Python + CUDA Fortran Workflow)"
echo "======================================================================="
echo ""

# Compiler
NVFORTRAN=nvfortran

# Compilation flags
FLAGS="-cuda -O3 -lcublas -lcudart -lcudnn -lcurand"

# Source file
SOURCE="cifar10_cudnn_v27.cuf"

# Output binary
OUTPUT="cifar10_cudnn_v27"

echo "Compiler: $NVFORTRAN"
echo "Source:   $SOURCE"
echo "Output:   $OUTPUT"
echo "Workflow: Python (data) + CUDA Fortran (training)"
echo "Flags:    $FLAGS"
echo ""

# Compile
echo "Compiling..."
$NVFORTRAN $FLAGS $SOURCE -o $OUTPUT

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "Compilation successful!"
    echo "======================================================================="
    echo ""
    echo "To run:"
    echo "  1. Prepare data: python prepare_cifar10_v27.py"
    echo "  2. Train model: ./$OUTPUT"
    echo ""
    echo "Data files expected in: cifar10_data/"
    echo ""
else
    echo ""
    echo "======================================================================="
    echo "Compilation failed!"
    echo "======================================================================="
    echo ""
    exit 1
fi
