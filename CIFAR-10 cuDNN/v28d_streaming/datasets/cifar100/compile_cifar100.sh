#!/bin/bash
#===============================================================================
# CIFAR-100 v28d Training Compilation Script
#===============================================================================
# Compiles CIFAR-100 training with streaming support
#
# Usage:
#   ./cifar100_train              # Full RAM mode
#   ./cifar100_train --stream     # Streaming mode
#===============================================================================

set -e

echo "================================================================================"
echo "Compiling CIFAR-100 Training (v28d - Streaming Support)"
echo "================================================================================"

if ! command -v nvfortran &> /dev/null; then
    echo "ERROR: nvfortran not found"
    exit 1
fi

echo "Detecting GPU compute capability..."
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ -z "$GPU_CC" ]; then
    GPU_CC=80
fi
echo "Detected GPU compute capability: $GPU_CC"

COMMON_DIR="../../common"
COMPILER="nvfortran"
FLAGS="-cuda -mp -O3 -lcublas -lcudart -lcudnn -lcurand"
OUTPUT="cifar100_train"

COMMON_SOURCES=(
    "${COMMON_DIR}/streaming_data_loader.cuf"
    "${COMMON_DIR}/cmdline_args.cuf"
    "${COMMON_DIR}/warp_shuffle.cuf"
    "${COMMON_DIR}/random_utils.cuf"
    "${COMMON_DIR}/adam_optimizer.cuf"
    "${COMMON_DIR}/gpu_batch_extraction.cuf"
    "${COMMON_DIR}/cuda_utils.cuf"
    "${COMMON_DIR}/model_export.cuf"
)

DATASET_SOURCES=(
    "cifar100_config.cuf"
)

echo ""
echo "Checking source files..."
for src in "${COMMON_SOURCES[@]}" "${DATASET_SOURCES[@]}"; do
    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        exit 1
    fi
    echo "  $src"
done

if [ -f "cifar100_main.cuf" ]; then
    echo "  cifar100_main.cuf"
fi

CMD="$COMPILER $FLAGS"
for src in "${COMMON_SOURCES[@]}" "${DATASET_SOURCES[@]}"; do
    CMD="$CMD $src"
done

if [ -f "cifar100_main.cuf" ]; then
    CMD="$CMD cifar100_main.cuf"
fi
CMD="$CMD -o $OUTPUT"

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
    echo "  ./$OUTPUT              # Full RAM mode"
    echo "  ./$OUTPUT --stream     # Streaming mode"
    echo "================================================================================"
else
    echo "Compilation failed"
    exit 1
fi
