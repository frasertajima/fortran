#!/bin/bash
#===============================================================================
# CIFAR-10 v28d Training Compilation Script
#===============================================================================
# Compiles CIFAR-10 training with:
#   - MANAGED MEMORY for large datasets
#   - WARP SHUFFLE for GPU-optimized loss/accuracy
#   - STREAMING support for datasets larger than RAM (optional at runtime)
#
# Directory structure:
#   ../../common/                - Shared modules
#   ./cifar10_config_large.cuf   - Large dataset config
#   ./cifar10_main.cuf           - Main training program
#
# Output: ./cifar10_train
#
# Runtime options:
#   ./cifar10_train              - Normal training (full RAM mode)
#   ./cifar10_train --stream     - Streaming mode (for huge datasets)
#===============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "Compiling CIFAR-10 Training (v28d - Streaming Support)"
echo "================================================================================"

# Check for nvfortran
if ! command -v nvfortran &> /dev/null; then
    echo "ERROR: nvfortran not found"
    echo "   Please install NVIDIA HPC SDK and add it to PATH"
    exit 1
fi

# Detect GPU compute capability
echo "Detecting GPU compute capability..."
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ -z "$GPU_CC" ]; then
    echo "WARNING: Could not detect GPU, using default cc80"
    GPU_CC=80
else
    echo "Detected GPU compute capability: $GPU_CC"
fi

# Compilation settings
COMMON_DIR="../../common"
COMPILER="nvfortran"
FLAGS="-cuda -mp -O3 -lcublas -lcudart -lcudnn -lcurand"
OUTPUT="cifar10_train"

# Source files (in dependency order - streaming first!)
COMMON_SOURCES=(
    "${COMMON_DIR}/streaming_data_loader.cuf"  # v28d: Streaming support
    "${COMMON_DIR}/cmdline_args.cuf"           # v28d: Command line parsing
    "${COMMON_DIR}/warp_shuffle.cuf"
    "${COMMON_DIR}/random_utils.cuf"
    "${COMMON_DIR}/adam_optimizer.cuf"
    "${COMMON_DIR}/gpu_batch_extraction.cuf"
    "${COMMON_DIR}/cuda_utils.cuf"
    "${COMMON_DIR}/model_export.cuf"
)

DATASET_SOURCES=(
    "cifar10_config_large.cuf"
)

# Check that source files exist
echo ""
echo "Checking source files..."
for src in "${COMMON_SOURCES[@]}"; do
    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        exit 1
    fi
    echo "  $src"
done

for src in "${DATASET_SOURCES[@]}"; do
    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        exit 1
    fi
    echo "  $src"
done

if [ -f "cifar10_main.cuf" ]; then
    echo "  cifar10_main.cuf"
fi

echo ""
echo "Compilation settings:"
echo "  Compiler: $COMPILER"
echo "  Flags:    $FLAGS"
echo "  Output:   $OUTPUT"

# Build compilation command
CMD="$COMPILER $FLAGS"

# Add all source files
for src in "${COMMON_SOURCES[@]}" "${DATASET_SOURCES[@]}"; do
    CMD="$CMD $src"
done

# Add main file
if [ -f "cifar10_main.cuf" ]; then
    CMD="$CMD cifar10_main.cuf"
fi

CMD="$CMD -o $OUTPUT"

# Compile
echo ""
echo "Compiling..."
echo "$ $CMD"
echo ""

if $CMD; then
    echo ""
    echo "================================================================================"
    echo "Compilation successful!"
    echo "================================================================================"
    echo ""
    echo "Usage:"
    echo "  ./$OUTPUT              # Full RAM mode (loads entire dataset)"
    echo "  ./$OUTPUT --stream     # Streaming mode (for huge datasets)"
    echo ""
    echo "Streaming mode benefits:"
    echo "  - Train on datasets larger than GPU+system RAM"
    echo "  - Memory usage stays constant (~3 MB) regardless of dataset size"
    echo "  - Double-buffered async I/O hides disk latency"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "Compilation failed"
    echo "================================================================================"
    exit 1
fi
