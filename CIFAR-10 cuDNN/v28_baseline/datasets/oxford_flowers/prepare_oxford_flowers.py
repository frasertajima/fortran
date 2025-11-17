"""
Extract MobileNetV2 features for Oxford Flowers 102 dataset
Exports to binary files compatible with CIFAR-10 CUDA Fortran infrastructure

This creates a generic interface between PyTorch pretrained models and custom CUDA Fortran training.
The same approach works for ANY dataset - just change the dataset loader!

Output format (matching CIFAR-10):
- features_train.bin: (1020, 1280) float32 - training features
- labels_train.bin: (1020,) float32 - training labels (0-101)
- features_val.bin: (1020, 1280) float32 - validation features
- labels_val.bin: (1020,) float32 - validation labels
- features_test.bin: (6149, 1280) float32 - test features
- labels_test.bin: (6149,) float32 - test labels

Your CUDA Fortran code just needs to:
1. Load these binary files (same format as CIFAR-10!)
2. Implement Dense(1280 → 102) with your custom regularization
3. Train and beat PyTorch!
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = './data/oxford_flowers'
OUTPUT_DIR = './oxford_features'
IMAGE_SIZE = 224
BATCH_SIZE = 32

if __name__ == '__main__':
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('MobileNetV2 Feature Extraction for Oxford Flowers 102')
    print('=' * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms (matching training script - simple [0,1] normalization)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # [0, 1] normalization
    ])
    
    # Load datasets
    print('\nLoading datasets...')
    train_dataset = datasets.Flowers102(
        root=DATA_DIR,
        split='train',
        transform=transform,
        download=True
    )
    
    val_dataset = datasets.Flowers102(
        root=DATA_DIR,
        split='val',
        transform=transform,
        download=True
    )
    
    test_dataset = datasets.Flowers102(
        root=DATA_DIR,
        split='test',
        transform=transform,
        download=True
    )
    
    print(f'  Train: {len(train_dataset)} samples')
    print(f'  Val: {len(val_dataset)} samples')
    print(f'  Test: {len(test_dataset)} samples')
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Load pretrained MobileNetV2 and extract features
    print('\nLoading MobileNetV2 feature extractor...')
    mobilenet = models.mobilenet_v2(pretrained=True)
    # Remove classifier, keep only features
    feature_extractor = mobilenet.features
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Add global average pooling (MobileNetV2 uses this before classifier)
    global_pool = nn.AdaptiveAvgPool2d(1)
    global_pool = global_pool.to(device)
    
    print('Feature extractor loaded (1280-dimensional output)')
    print(f'  Frozen parameters: {sum(p.numel() for p in feature_extractor.parameters()):,}')
    
    
    def extract_features(data_loader, split_name):
        """Extract features for a dataset split"""
        features_list = []
        labels_list = []
    
        print(f'\nExtracting {split_name} features...')
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=split_name):
                images = images.to(device)
    
                # Extract features
                x = feature_extractor(images)
                x = global_pool(x)
                x = torch.flatten(x, 1)  # (batch_size, 1280)
    
                features_list.append(x.cpu().numpy())
                labels_list.append(labels.numpy())
    
        # Concatenate all batches
        features = np.concatenate(features_list, axis=0).astype(np.float32)
        labels = np.concatenate(labels_list, axis=0).astype(np.float32)
    
        print(f'  Features shape: {features.shape}')
        print(f'  Labels shape: {labels.shape}')
        print(f'  Features range: [{features.min():.4f}, {features.max():.4f}]')
        print(f'  Labels range: [{labels.min():.0f}, {labels.max():.0f}]')
    
        return features, labels
    
    
    # Extract features for all splits
    train_features, train_labels = extract_features(train_loader, 'train')
    val_features, val_labels = extract_features(val_loader, 'validation')
    test_features, test_labels = extract_features(test_loader, 'test')
    
    # Verify dimensions
    assert train_features.shape == (1020, 1280), f"Train features shape mismatch: {train_features.shape}"
    assert val_features.shape == (1020, 1280), f"Val features shape mismatch: {val_features.shape}"
    assert test_features.shape == (6149, 1280), f"Test features shape mismatch: {test_features.shape}"
    
    # Save to binary files (Fortran-compatible format)
    print('\nSaving binary files...')
    
    # Training data
    train_features_path = Path(OUTPUT_DIR) / 'features_train.bin'
    train_labels_path = Path(OUTPUT_DIR) / 'labels_train.bin'
    train_features.tofile(train_features_path)
    train_labels.tofile(train_labels_path)
    print(f'  ✓ {train_features_path}')
    print(f'  ✓ {train_labels_path}')
    
    # Validation data
    val_features_path = Path(OUTPUT_DIR) / 'features_val.bin'
    val_labels_path = Path(OUTPUT_DIR) / 'labels_val.bin'
    val_features.tofile(val_features_path)
    val_labels.tofile(val_labels_path)
    print(f'  ✓ {val_features_path}')
    print(f'  ✓ {val_labels_path}')
    
    # Test data
    test_features_path = Path(OUTPUT_DIR) / 'features_test.bin'
    test_labels_path = Path(OUTPUT_DIR) / 'labels_test.bin'
    test_features.tofile(test_features_path)
    test_labels.tofile(test_labels_path)
    print(f'  ✓ {test_features_path}')
    print(f'  ✓ {test_labels_path}')
    
    # Verification statistics
    print('\n' + '=' * 70)
    print('Feature Extraction Complete!')
    print('=' * 70)
    print('\nDataset Statistics:')
    print(f'  Training:   {len(train_features):5,} samples × 1280 features')
    print(f'  Validation: {len(val_features):5,} samples × 1280 features')
    print(f'  Test:       {len(test_features):5,} samples × 1280 features')
    
    print('\nBinary File Sizes:')
    train_size = train_features.nbytes + train_labels.nbytes
    val_size = val_features.nbytes + val_labels.nbytes
    test_size = test_features.nbytes + test_labels.nbytes
    print(f'  Training:   {train_size / 1024 / 1024:.2f} MB')
    print(f'  Validation: {val_size / 1024 / 1024:.2f} MB')
    print(f'  Test:       {test_size / 1024 / 1024:.2f} MB')
    print(f'  Total:      {(train_size + val_size + test_size) / 1024 / 1024:.2f} MB')
    
    print('\nFor Your CUDA Fortran Code:')
    print('  1. Read features_*.bin as (N, 1280) float32 arrays')
    print('  2. Read labels_*.bin as (N,) float32 arrays')
    print('  3. Implement Dense(1280 → 102) layer')
    print('  4. Train with your custom regularization!')
    print('  5. Target: Beat PyTorch\'s 76.26% test accuracy!')
    
    print('\n✅ Ready for CUDA Fortran implementation!')
