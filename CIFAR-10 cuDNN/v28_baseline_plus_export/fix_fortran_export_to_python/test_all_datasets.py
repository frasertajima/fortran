"""
Comprehensive Accuracy Test for All Datasets
Tests PyTorch inference with Fortran-trained weights.
"""

import sys
sys.path.insert(0, 'inference')

import torch
import numpy as np
from pathlib import Path
from model_loader import load_v28_model

def test_dataset(dataset_name, model_dir, data_dir, in_channels, num_classes, input_size, num_test_samples):
    """Test a single dataset"""
    print("="*70)
    print(f"Testing {dataset_name}")
    print("="*70)
    
    model_path = Path(model_dir)
    data_path = Path(data_dir)
    
    # Check if model exists
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_dir}")
        print(f"   Skipping {dataset_name}")
        print("="*70)
        print()
        return None
    
    try:
        # Load model
        model = load_v28_model(model_dir, in_channels=in_channels, num_classes=num_classes, device='cpu')
        model.eval()
        
        # Load test data
        test_images = np.fromfile(data_path / 'images_test.bin', dtype=np.float32)
        test_labels = np.fromfile(data_path / 'labels_test.bin', dtype=np.int32)
        
        # Reshape: Fortran exports as (features, N), need to transpose to (N, features)
        features = in_channels * input_size * input_size
        test_images = test_images.reshape((features, num_test_samples), order='C').T
        
        print(f"Loaded {num_test_samples} test samples")
        print(f"Running inference...")
        
        # Test
        correct = 0
        with torch.no_grad():
            for i in range(num_test_samples):
                # Reshape to (C, H, W)
                img = test_images[i].reshape((in_channels, input_size, input_size))
                img_tensor = torch.from_numpy(img).unsqueeze(0)
                
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
                
                if pred == test_labels[i]:
                    correct += 1
                
                if (i + 1) % 1000 == 0:
                    acc = 100.0 * correct / (i + 1)
                    print(f"  {i+1}/{num_test_samples}: {acc:.2f}%")
        
        accuracy = 100.0 * correct / num_test_samples
        print()
        print(f"Final Accuracy: {accuracy:.2f}% ({correct}/{num_test_samples})")
        print("="*70)
        print()
        
        return accuracy
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*70)
        print()
        return None

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE ACCURACY TEST - ALL DATASETS")
    print("="*70)
    print()
    
    results = {}
    
    # CIFAR-10
    results['CIFAR-10'] = test_dataset(
        'CIFAR-10',
        'datasets/cifar10/saved_models/cifar10',
        'datasets/cifar10/cifar10_data',
        in_channels=3,
        num_classes=10,
        input_size=32,
        num_test_samples=10000
    )
    
    # CIFAR-100
    results['CIFAR-100'] = test_dataset(
        'CIFAR-100',
        'datasets/cifar100/saved_models/cifar100',
        'datasets/cifar100/cifar100_data',
        in_channels=3,
        num_classes=100,
        input_size=32,
        num_test_samples=10000
    )
    
    # SVHN
    results['SVHN'] = test_dataset(
        'SVHN',
        'datasets/svhn/saved_models/svhn',
        'datasets/svhn/svhn_data',
        in_channels=3,
        num_classes=10,
        input_size=32,
        num_test_samples=26032
    )
    
    # Fashion-MNIST
    results['Fashion-MNIST'] = test_dataset(
        'Fashion-MNIST',
        'datasets/fashion_mnist/saved_models/fashion_mnist',
        'datasets/fashion_mnist/fashion_mnist_data',
        in_channels=1,
        num_classes=10,
        input_size=28,
        num_test_samples=10000
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - ALL DATASETS")
    print("="*70)
    tested = 0
    for dataset, acc in results.items():
        if acc is not None:
            tested += 1
            status = "✅" if acc > 70 else "⚠️"
            print(f"{status} {dataset:20s}: {acc:6.2f}%")
        else:
            print(f"⚠️  {dataset:20s}: Not tested (model not found)")
    print("="*70)
    print(f"\nTested: {tested}/{len(results)} datasets")
    print("="*70)

if __name__ == "__main__":
    main()
