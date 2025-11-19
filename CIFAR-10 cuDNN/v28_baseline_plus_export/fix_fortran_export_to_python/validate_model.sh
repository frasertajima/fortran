#!/bin/bash
#===============================================================================
# v28 Baseline Model Validation Script
#===============================================================================
# Validates model export files after training to catch bugs early
#
# Usage:
#   bash validate_model.sh cifar10
#   bash validate_model.sh cifar100
#   bash validate_model.sh svhn
#   bash validate_model.sh fashion_mnist
#
# Checks:
#   1. saved_models directory exists
#   2. Exactly 19 .bin files + 1 metadata file
#   3. fc1_weights.bin has correct size for dataset
#
# Exit codes:
#   0 = All checks passed
#   1 = Validation failed
#===============================================================================

set -e

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: bash validate_model.sh <dataset>"
    echo ""
    echo "Available datasets:"
    echo "  cifar10"
    echo "  cifar100"
    echo "  svhn"
    echo "  fashion_mnist"
    exit 1
fi

echo "================================================================================"
echo "Validating v28 Model Export: $DATASET"
echo "================================================================================"
echo ""

# 1. Check saved_models directory exists
SAVED_DIR="datasets/$DATASET/saved_models/$DATASET"

if [ ! -d "$SAVED_DIR" ]; then
    echo "❌ ERROR: saved_models directory not found!"
    echo "   Expected: $SAVED_DIR"
    echo ""
    echo "   This means:"
    echo "   - Model was not trained yet, OR"
    echo "   - Training completed but export failed"
    echo ""
    echo "   Fix:"
    echo "   1. Check compile_*.sh includes model_export.cuf"
    echo "   2. Check *_main.cuf has 'use model_export'"
    echo "   3. Check *_main.cuf calls export_model_generic()"
    exit 1
fi

echo "✅ saved_models directory exists: $SAVED_DIR"
echo ""

# 2. Check file count
BIN_COUNT=$(ls "$SAVED_DIR"/*.bin 2>/dev/null | wc -l)
METADATA_COUNT=$(ls "$SAVED_DIR"/model_metadata.txt 2>/dev/null | wc -l)

echo "File count check:"
echo "  Binary files (.bin): $BIN_COUNT / 19 expected"
echo "  Metadata file: $METADATA_COUNT / 1 expected"
echo ""

if [ "$BIN_COUNT" != "19" ]; then
    echo "❌ ERROR: Expected 19 .bin files, found $BIN_COUNT"
    echo ""
    echo "   Expected files:"
    echo "   - conv1_weights.bin, conv1_bias.bin"
    echo "   - conv2_weights.bin, conv2_bias.bin"
    echo "   - conv3_weights.bin, conv3_bias.bin"
    echo "   - fc1_weights.bin, fc1_bias.bin"
    echo "   - fc2_weights.bin, fc2_bias.bin"
    echo "   - fc3_weights.bin, fc3_bias.bin"
    echo "   - bn1_scale.bin, bn1_bias.bin, bn1_running_mean.bin, bn1_running_var.bin"
    echo "   - bn2_scale.bin, bn2_bias.bin, bn2_running_mean.bin, bn2_running_var.bin"
    echo "   - bn3_scale.bin, bn3_bias.bin, bn3_running_mean.bin, bn3_running_var.bin"
    echo ""
    echo "   Missing files found:"
    ls "$SAVED_DIR"/*.bin 2>/dev/null || echo "   (none)"
    exit 1
fi

if [ "$METADATA_COUNT" != "1" ]; then
    echo "❌ ERROR: model_metadata.txt not found"
    exit 1
fi

echo "✅ All 20 files present (19 .bin + 1 metadata.txt)"
echo ""

# 3. Check fc1_weights.bin size (critical for flatten_size validation)
FC1_PATH="$SAVED_DIR/fc1_weights.bin"

# Get file size (works on both macOS and Linux)
if command -v stat &> /dev/null; then
    # Try macOS format first
    FC1_SIZE=$(stat -f%z "$FC1_PATH" 2>/dev/null || stat -c%s "$FC1_PATH" 2>/dev/null)
else
    # Fallback to ls
    FC1_SIZE=$(ls -l "$FC1_PATH" | awk '{print $5}')
fi

FC1_SIZE_MB=$((FC1_SIZE / 1024 / 1024))

echo "fc1_weights.bin size check:"
echo "  File size: $FC1_SIZE bytes ($FC1_SIZE_MB MB)"
echo ""

# Expected sizes based on flatten_size
# Fashion-MNIST (28×28): 128 * 3 * 3 = 1152 → 512 * 1152 * 4 = 2,359,296 bytes (2.25 MB)
# CIFAR-10/100/SVHN (32×32): 128 * 4 * 4 = 2048 → 512 * 2048 * 4 = 4,194,304 bytes (4.0 MB)

if [ "$DATASET" == "fashion_mnist" ]; then
    EXPECTED_SIZE=2359296  # 512 * 1152 * 4
    EXPECTED_MB=2.25
    EXPECTED_FLATTEN=1152

    if [ "$FC1_SIZE" != "$EXPECTED_SIZE" ]; then
        echo "❌ ERROR: fc1_weights.bin has WRONG size!"
        echo ""
        echo "   Expected: $EXPECTED_SIZE bytes ($EXPECTED_MB MB) for Fashion-MNIST"
        echo "   Got:      $FC1_SIZE bytes ($FC1_SIZE_MB MB)"
        echo ""
        echo "   This indicates:"
        echo "   - Model was trained with HARDCODED flatten_size bug!"
        echo "   - flatten_size = 2048 (wrong) instead of 1152 (correct)"
        echo "   - Model learned from UNINITIALIZED MEMORY (garbage)"
        echo ""
        echo "   🚨 CRITICAL BUG: Model is unusable and must be RETRAINED"
        echo ""
        echo "   Fix:"
        echo "   1. Check fashion_mnist_main.cuf uses parameterized flatten_size"
        echo "   2. Verify: grep 'CONV3_FILTERS \* 4 \* 4' should return ZERO matches"
        echo "   3. All allocations should use: CONV3_FILTERS * ((INPUT_HEIGHT/4)/2) * ((INPUT_WIDTH/4)/2)"
        echo "   4. Recompile and retrain from scratch"
        exit 1
    fi

    echo "✅ fc1_weights.bin size correct: $EXPECTED_MB MB (flatten_size=$EXPECTED_FLATTEN)"
else
    # CIFAR-10, CIFAR-100, SVHN (all 32×32)
    EXPECTED_SIZE=4194304  # 512 * 2048 * 4
    EXPECTED_MB=4.0
    EXPECTED_FLATTEN=2048

    if [ "$FC1_SIZE" != "$EXPECTED_SIZE" ]; then
        echo "❌ ERROR: fc1_weights.bin has WRONG size!"
        echo ""
        echo "   Expected: $EXPECTED_SIZE bytes ($EXPECTED_MB MB) for $DATASET"
        echo "   Got:      $FC1_SIZE bytes ($FC1_SIZE_MB MB)"
        echo ""
        echo "   This indicates a flatten_size bug!"
        exit 1
    fi

    echo "✅ fc1_weights.bin size correct: $EXPECTED_MB MB (flatten_size=$EXPECTED_FLATTEN)"
fi

echo ""
echo "================================================================================"
echo "✅ Model Export Validation PASSED!"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Dataset: $DATASET"
echo "  Files: 19 .bin files + metadata"
echo "  fc1_weights: Correct size for dataset"
echo ""
echo "Next steps:"
echo "  1. Load model in Python:"
echo "     cd datasets/$DATASET/"
echo "     jupyter notebook ${DATASET}_inference.ipynb"
echo ""
echo "  2. Verify inference accuracy:"
echo "     - Test accuracy should be > 85% (cifar10/svhn/fashion_mnist)"
echo "     - Test accuracy should be > 50% (cifar100)"
echo "     - Per-class accuracy should be balanced"
echo "     - Predictions should be distributed across all classes"
echo ""
echo "  3. Red flags (model might be broken):"
echo "     - Test accuracy < 50%"
echo "     - Predicting one class 70%+ of the time"
echo "     - High confidence + wrong predictions"
echo ""
echo "================================================================================"
