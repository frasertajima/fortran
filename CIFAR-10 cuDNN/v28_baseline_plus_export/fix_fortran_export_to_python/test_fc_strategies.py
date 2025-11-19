"""
Test different FC weight loading strategies.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load test data
test_data_file = Path("datasets/cifar10/cifar10_data/images_test.bin")
test_labels_file = Path("datasets/cifar10/cifar10_data/labels_test.bin")

test_data = np.fromfile(test_data_file, dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T

test_labels = np.fromfile(test_labels_file, dtype=np.int32)

# Use first 100 images
test_data_small = test_data[:100]
test_labels_small = test_labels[:100]

# Load conv and BN weights (these should be correct)
model_dir = Path("datasets/cifar10/saved_models/cifar10")

print("Testing different FC weight loading strategies...")
print()

strategies = [
    {"name": "Strategy 1: F-order reshape, no transpose", "order": "F", "transpose": False},
    {"name": "Strategy 2: F-order reshape, then transpose", "order": "F", "transpose": True},
    {"name": "Strategy 3: C-order reshape, no transpose", "order": "C", "transpose": False},
    {"name": "Strategy 4: C-order reshape, then transpose", "order": "C", "transpose": True},
]

for strategy in strategies:
    print(f"Testing: {strategy['name']}")
    
    try:
        # Simple model for testing
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
                self.bn1 = nn.BatchNorm2d(32)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(2048, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 10)
                self.relu = nn.LeakyReLU(0.01)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = self.pool(self.relu(self.bn1(self.conv1(x))))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                return x
        
        model = TestModel()
        model.eval()
        
        # Load conv1 (we know this works with F-order)
        conv1_w = np.fromfile(model_dir / 'conv1_weights.bin', dtype=np.float32)
        conv1_w = conv1_w.reshape((32, 3, 3, 3), order='F')
        conv1_w = np.ascontiguousarray(conv1_w)
        conv1_b = np.fromfile(model_dir / 'conv1_bias.bin', dtype=np.float32)
        model.conv1.weight.data = torch.from_numpy(conv1_w)
        model.conv1.bias.data = torch.from_numpy(conv1_b)
        
        # Load BN1
        bn1_scale = np.fromfile(model_dir / 'bn1_scale.bin', dtype=np.float32)
        bn1_bias = np.fromfile(model_dir / 'bn1_bias.bin', dtype=np.float32)
        bn1_mean = np.fromfile(model_dir / 'bn1_running_mean.bin', dtype=np.float32)
        bn1_var = np.fromfile(model_dir / 'bn1_running_var.bin', dtype=np.float32)
        model.bn1.weight.data = torch.from_numpy(bn1_scale)
        model.bn1.bias.data = torch.from_numpy(bn1_bias)
        model.bn1.running_mean.data = torch.from_numpy(bn1_mean)
        model.bn1.running_var.data = torch.from_numpy(bn1_var)
        
        # Load FC weights with this strategy
        fc1_data = np.fromfile(model_dir / 'fc1_weights.bin', dtype=np.float32)
        fc1_w = fc1_data.reshape((512, 2048), order=strategy['order'])
        if strategy['transpose']:
            fc1_w = fc1_w.T
        fc1_w = np.ascontiguousarray(fc1_w)
        fc1_b = np.fromfile(model_dir / 'fc1_bias.bin', dtype=np.float32)
        model.fc1.weight.data = torch.from_numpy(fc1_w)
        model.fc1.bias.data = torch.from_numpy(fc1_b)
        
        fc2_data = np.fromfile(model_dir / 'fc2_weights.bin', dtype=np.float32)
        fc2_w = fc2_data.reshape((256, 512), order=strategy['order'])
        if strategy['transpose']:
            fc2_w = fc2_w.T
        fc2_w = np.ascontiguousarray(fc2_w)
        fc2_b = np.fromfile(model_dir / 'fc2_bias.bin', dtype=np.float32)
        model.fc2.weight.data = torch.from_numpy(fc2_w)
        model.fc2.bias.data = torch.from_numpy(fc2_b)
        
        fc3_data = np.fromfile(model_dir / 'fc3_weights.bin', dtype=np.float32)
        fc3_w = fc3_data.reshape((10, 256), order=strategy['order'])
        if strategy['transpose']:
            fc3_w = fc3_w.T
        fc3_w = np.ascontiguousarray(fc3_w)
        fc3_b = np.fromfile(model_dir / 'fc3_bias.bin', dtype=np.float32)
        model.fc3.weight.data = torch.from_numpy(fc3_w)
        model.fc3.bias.data = torch.from_numpy(fc3_b)
        
        # Test
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
        print(f"  Accuracy: {accuracy:.2f}%")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        print()

print("Expected: ~78-79% accuracy")
