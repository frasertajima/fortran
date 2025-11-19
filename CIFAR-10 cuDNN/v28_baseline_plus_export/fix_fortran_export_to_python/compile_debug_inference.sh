#!/bin/bash
# Compile debug inference program

echo "Compiling debug inference program..."
nvfortran -cuda -O2 \
    common/model_export.cuf \
    debug_inference.cuf \
    -o debug_inference

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "Run with: ./debug_inference"
else
    echo "❌ Compilation failed!"
    exit 1
fi
