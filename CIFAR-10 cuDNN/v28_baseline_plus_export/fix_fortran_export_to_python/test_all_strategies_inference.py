"""
Debug: Let's go back to basics and try ALL strategies on the FULL model inference.

We'll test each strategy and see which one gives the best accuracy on a small test set.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load test data
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_labels_file = Path("datasets/cifar10/cifar10_data/labels_test.bin")

test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T  # (10000, 3072)

test_labels = np.fromfile(test_labels_file, dtype=np.int32)

# Use first 100 images for quick testing
test_data_small = test_data[:100]
test_labels_small = test_labels[:100]

print("Testing different weight loading strategies on 100 test images...")
print()

# Define strategies to test
strategies = [
    {
        'name': 'Strategy 1: Reshape (32,3,3,3) F-order, no transpose',
        'conv1_shape': (32, 3, 3, 3),
        'conv1_transpose': None,
        'conv2_shape': (64, 3, 3, 32),
        'conv2_transpose': None,
        'conv3_shape': (128, 3, 3, 64),
        'conv3_transpose': None,
    },
    {
        'name': 'Strategy 2: Reshape (3,3,3,32) F-order, transpose (3,2,0,1)',
        'conv1_shape': (3, 3, 3, 32),
        'conv1_transpose': (3, 2, 0, 1),
        'conv2_shape': (3, 3, 32, 64),
        'conv2_transpose': (3, 2, 0, 1),
        'conv3_shape': (3, 3, 64, 128),
        'conv3_transpose': (3, 2, 0, 1),
    },
    {
        'name': 'Strategy 3: Reshape (3,3,32,3) F-order, transpose (2,3,0,1)',
        'conv1_shape': (3, 3, 32, 3),
        'conv1_transpose': (2, 3, 0, 1),
        'conv2_shape': (3, 3, 64, 32),
        'conv2_transpose': (2, 3, 0, 1),
        'conv3_shape': (3, 3, 128, 64),
        'conv3_transpose': (2, 3, 0, 1),
    },
    {
        'name': 'Strategy 4: Reshape (32,3,3,3) F-order, transpose (0,3,1,2)',
        'conv1_shape': (32, 3, 3, 3),
        'conv1_transpose': (0, 3, 1, 2),
        'conv2_shape': (64, 3, 3, 32),
        'conv2_transpose': (0, 3, 1, 2),
        'conv3_shape': (128, 3, 3, 64),
        'conv3_transpose': (0, 3, 1, 2),
    },
]

model_dir = Path("datasets/cifar10/saved_models/cifar10")

for strategy in strategies:
    print(f"Testing: {strategy['name']}")
    
    try:
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
                self.bn3 = nn.BatchNorm2d(128)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(2048, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 10)
                self.relu = nn.LeakyReLU(0.01)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = self.pool(self.relu(self.bn1(self.conv1(x))))
                x = self.pool(self.relu(self.bn2(self.conv2(x))))
                x = self.pool(self.relu(self.bn3(self.conv3(x))))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        # Load conv weights with this strategy
        for layer_name, shape_key, transpose_key in [
            ('conv1', 'conv1_shape', 'conv1_transpose'),
            ('conv2', 'conv2_shape', 'conv2_transpose'),
            ('conv3', 'conv3_shape', 'conv3_transpose'),
        ]:
            w_data = np.fromfile(model_dir / f'{layer_name}_weights.bin', dtype=np.float32)
            w = w_data.reshape(strategy[shape_key], order='F')
            if strategy[transpose_key] is not None:
                w = np.transpose(w, strategy[transpose_key])
            w = np.ascontiguousarray(w)
            
            b_data = np.fromfile(model_dir / f'{layer_name}_bias.bin', dtype=np.float32)
            
            getattr(model, layer_name).weight.data = torch.from_numpy(w)
            getattr(model, layer_name).bias.data = torch.from_numpy(b_data)
        
        # Load FC weights (these should be the same for all strategies)
        for fc_name in ['fc1', 'fc2', 'fc3']:
            w_data = np.fromfile(model_dir / f'{fc_name}_weights.bin', dtype=np.float32)
            if fc_name == 'fc1':
                w = w_data.reshape((512, 2048), order='C').T
            elif fc_name == 'fc2':
                w = w_data.reshape((256, 512), order='C').T
            else:
                w = w_data.reshape((10, 256), order='C').T
            
            b_data = np.fromfile(model_dir / f'{fc_name}_bias.bin', dtype=np.float32)
            
            getattr(model, fc_name).weight.data = torch.from_numpy(np.ascontiguousarray(w))
            getattr(model, fc_name).bias.data = torch.from_numpy(b_data)
        
        # Load BN weights
        for bn_name in ['bn1', 'bn2', 'bn3']:
            scale = np.fromfile(model_dir / f'{bn_name}_scale.bin', dtype=np.float32)
            bias = np.fromfile(model_dir / f'{bn_name}_bias.bin', dtype=np.float32)
            mean = np.fromfile(model_dir / f'{bn_name}_running_mean.bin', dtype=np.float32)
            var = np.fromfile(model_dir / f'{bn_name}_running_var.bin', dtype=np.float32)
            
            getattr(model, bn_name).weight.data = torch.from_numpy(scale)
            getattr(model, bn_name).bias.data = torch.from_numpy(bias)
            getattr(model, bn_name).running_mean.data = torch.from_numpy(mean)
            getattr(model, bn_name).running_var.data = torch.from_numpy(var)
        
        # Test on small dataset
        correct = 0
        with torch.no_grad():
            for i in range(len(test_data_small)):
                img = test_data_small[i].reshape((3, 32, 32))
                img_tensor = torch.from_numpy(img).unsqueeze(0)
                
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
                
                if pred == test_labels_small[i]:
                    correct += 1
        
        accuracy = 100.0 * correct / len(test_data_small)
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{len(test_data_small)})")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        print()

print("Expected: ~78-79% accuracy (matching Fortran training)")
