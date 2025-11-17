#!/bin/bash
#===============================================================================
# Oxford Flowers v27 Compilation Script (Standalone)
#===============================================================================
# Compiles Oxford Flowers training using the proven v27 version
# This version is self-contained with all modules embedded
#
# Output: ./oxford_flowers_train
#===============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "Compiling Oxford Flowers Training (v28 Baseline)"
echo "================================================================================"

# Check for nvfortran
if ! command -v nvfortran &> /dev/null; then
    echo "❌ ERROR: nvfortran not found"
    echo "   Please install NVIDIA HPC SDK and add it to PATH"
    echo ""
    echo "   Example:"
    echo "   export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin:\$PATH"
    exit 1
fi

# Detect GPU compute capability
echo "Detecting GPU compute capability..."
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ -z "$GPU_CC" ]; then
    echo "⚠️  WARNING: Could not detect GPU, using default cc80 (Ampere)"
    GPU_CC=80
else
    echo "✅ Detected GPU compute capability: $GPU_CC"
fi

# Compilation settings (v27 standalone - no external modules needed)
COMPILER="nvfortran"
FLAGS="-cuda -O3 -lcublas -lcudart -lcurand"
OUTPUT="oxford_flowers_train"

# Check that main file exists
echo ""
echo "Checking source files..."
if [ ! -f "oxford_flowers_main.cuf" ]; then
    echo "❌ ERROR: oxford_flowers_main.cuf not found"
    exit 1
fi
echo "  ✓ oxford_flowers_main.cuf (v27 standalone)"
echo ""
echo "Compilation settings:"
echo "  Compiler: $COMPILER"
echo "  Flags:    $FLAGS"
echo "  Output:   $OUTPUT"

# Build compilation command (standalone v27 - all modules embedded)
CMD="$COMPILER $FLAGS oxford_flowers_main.cuf -o $OUTPUT"

# Compile
echo ""
echo "Compiling..."
echo "$ $CMD"
echo ""

if $CMD; then
    echo ""
    echo "================================================================================"
    echo "✅ Compilation successful!"
    echo "================================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Prepare data:   python prepare_oxford_flowers.py"
    echo "  2. Train model:    ./$OUTPUT"
    echo ""
    echo "Expected performance:"
    echo "  - Accuracy: ~78-79% (matches PyTorch)"
    echo "  - Time: ~1 second for 100 epochs (very fast!)"
    echo "  - Memory: Minimal (dense layer only)"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "❌ Compilation failed"
    echo "================================================================================"
    echo ""
    echo "Common issues:"
    echo "  - Missing CUDA libraries: Ensure -lcublas -lcudart are available"
    echo "  - Module conflicts: Remove *.mod files and recompile"
    echo ""
    exit 1
fi
