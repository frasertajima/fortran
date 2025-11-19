#!/bin/bash
# Compile debug test program

echo "Compiling debug test program..."
nvfortran -cuda -O2 \
    common/model_export.cuf \
    debug_test.cuf \
    -o debug_test

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
else
    echo "❌ Compilation failed!"
    exit 1
fi
