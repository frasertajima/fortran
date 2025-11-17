#!/bin/bash
#===============================================================================
# SVHN v28 Baseline Compilation Script
#===============================================================================
# Compiles SVHN training using the modular v28 baseline framework
#
# Directory structure:
#   ../../common/        - Shared modules (optimizer, batch extraction, etc.)
#   ./svhn_config.cuf    - SVHN dataset configuration
#   ./svhn_main.cuf      - Main training program (to be created)
#
# Output: ./svhn_train
#===============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "Compiling SVHN Training (v28 Baseline - Modular Framework)"
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

# Compilation settings (matching old working v28)
COMMON_DIR="../../common"
COMPILER="nvfortran"
FLAGS="-cuda -O3 -lcublas -lcudart -lcudnn -lcurand"
OUTPUT="svhn_train"

# Source files (in dependency order)
COMMON_SOURCES=(
    "${COMMON_DIR}/random_utils.cuf"
    "${COMMON_DIR}/adam_optimizer.cuf"
    "${COMMON_DIR}/gpu_batch_extraction.cuf"
    "${COMMON_DIR}/cuda_utils.cuf"
)

DATASET_SOURCES=(
    "svhn_config.cuf"
)

# Check if main training file exists
if [ ! -f "svhn_main.cuf" ]; then
    echo ""
    echo "⚠️  WARNING: svhn_main.cuf not found"
    echo "   This file will be created in the next step."
    echo "   For now, testing compilation of modules only..."
    echo ""
fi

# Check that common modules exist
echo ""
echo "Checking source files..."
for src in "${COMMON_SOURCES[@]}"; do
    if [ ! -f "$src" ]; then
        echo "❌ ERROR: $src not found"
        exit 1
    fi
    echo "  ✓ $src"
done

for src in "${DATASET_SOURCES[@]}"; do
    if [ ! -f "$src" ]; then
        echo "❌ ERROR: $src not found"
        exit 1
    fi
    echo "  ✓ $src"
done

echo ""
echo "Compilation settings:"
echo "  Compiler: $COMPILER"
echo "  Flags:    $FLAGS"
echo "  Output:   $OUTPUT"

# Build compilation command (simple like old v28)
CMD="$COMPILER $FLAGS"

# Add all source files
for src in "${COMMON_SOURCES[@]}" "${DATASET_SOURCES[@]}"; do
    CMD="$CMD $src"
done

# Add main file if it exists
if [ -f "svhn_main.cuf" ]; then
    CMD="$CMD svhn_main.cuf"
fi

# Add output
CMD="$CMD -o $OUTPUT"

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
    if [ -f "svhn_main.cuf" ]; then
        echo ""
        echo "Next steps:"
        echo "  1. Prepare data:   python prepare_svhn.py"
        echo "  2. Train model:    ./$OUTPUT"
        echo ""
        echo "Expected performance:"
        echo "  - Accuracy: ~92-93% (street view digits)"
        echo "  - Time: ~80 seconds (15 epochs)"
        echo "  - Memory: ~1.0 GB GPU (more training samples)"
    else
        echo ""
        echo "⚠️  Main training file (svhn_main.cuf) not created yet."
        echo "   Modules compiled successfully, but cannot run training."
    fi
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "❌ Compilation failed"
    echo "================================================================================"
    echo ""
    echo "Common issues:"
    echo "  - Missing cuDNN: Install CUDA toolkit with cuDNN"
    echo "  - Missing CUDA libraries: Ensure -lcublas -lcudart -lcurand are available"
    echo "  - Module conflicts: Remove *.mod files and recompile"
    echo ""
    exit 1
fi
