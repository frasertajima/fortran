#!/bin/bash

# Script to comment out transpose_conv_weights_to_c_order in all datasets

echo "Commenting out transpose_conv_weights_to_c_order in all datasets..."

for dataset in svhn fashion_mnist cifar100; do
    echo "Processing $dataset..."
    
    file="datasets/$dataset/${dataset}_main.cuf"
    
    if [ -f "$file" ]; then
        # Create backup
        cp "$file" "$file.backup"
        
        # Use sed to comment out the transpose blocks
        # This is a simplified approach - for production, we'd want more careful editing
        echo "  ⚠️  Manual editing required for $dataset"
        echo "     File: $file"
        echo "     Search for: transpose_conv_weights_to_c_order"
        echo "     Comment out the blocks similar to CIFAR-10"
    else
        echo "  ❌ File not found: $file"
    fi
done

echo ""
echo "Summary:"
echo "- CIFAR-10: ✅ Already commented out"
echo "- SVHN: ⚠️  Needs manual editing"
echo "- Fashion-MNIST: ⚠️  Needs manual editing"  
echo "- CIFAR-100: ⚠️  Needs manual editing"
