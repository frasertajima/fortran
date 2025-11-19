"""
Train a PyTorch model, export weights, reload, and test.
This will verify the export/load cycle works correctly.
"""

import sys
sys.path.insert(0, 'inference')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from model_loader import V28_CNN, load_v28_model

print("="*70)
print("PyTorch Export/Load Cycle Test")
print("="*70)
print()

# Load CIFAR-10 data
print("Loading CIFAR-10 data...")
train_data = np.fromfile('datasets/cifar10/cifar10_data/images_train.bin', dtype=np.float32)
train_data = train_data.reshape((3072, 50000), order='C').T
train_labels = np.fromfile('datasets/cifar10/cifar10_data/labels_train.bin', dtype=np.int32)

test_data = np.fromfile('datasets/cifar10/cifar10_data/images_test.bin', dtype=np.float32)
test_data = test_data.reshape((3072, 10000), order='C').T
test_labels = np.fromfile('datasets/cifar10/cifar10_data/labels_test.bin', dtype=np.int32)

print(f"  Train: {train_data.shape}, Test: {test_data.shape}")
print()

# Create model
print("Creating model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = V28_CNN(in_channels=3, num_classes=10, input_size=32).to(device)
print(f"  Device: {device}")
print()

# Train for just 1 epoch (quick test)
print("Training for 1 epoch...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
batch_size = 128
n_batches = len(train_data) // batch_size

for batch_idx in range(min(100, n_batches)):  # Just 100 batches for speed
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    
    batch_data = train_data[start_idx:end_idx].reshape(batch_size, 3, 32, 32)
    batch_labels = train_labels[start_idx:end_idx]
    
    inputs = torch.from_numpy(batch_data).to(device)
    labels = torch.from_numpy(batch_labels).long().to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (batch_idx + 1) % 20 == 0:
        print(f"  Batch {batch_idx+1}/{min(100, n_batches)}, Loss: {loss.item():.4f}")

print()

# Test accuracy before export
print("Testing accuracy before export...")
model.eval()
correct = 0
n_test = 1000

with torch.no_grad():
    for i in range(n_test):
        img = test_data[i].reshape((3, 32, 32))
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
        if pred == test_labels[i]:
            correct += 1

accuracy_before = 100.0 * correct / n_test
print(f"  Accuracy before export: {accuracy_before:.2f}%")
print()

# Export weights in Fortran format
print("Exporting weights in Fortran F-order format...")
export_dir = Path('pytorch_export_test')
export_dir.mkdir(exist_ok=True)

def export_array(array, filename):
    """Export numpy array in F-order (like Fortran does)"""
    filepath = export_dir / filename
    # Convert to numpy if needed
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    # Write as binary
    array.astype(np.float32).tofile(filepath)
    print(f"  Exported {filename}: shape {array.shape}")

# Export conv weights
export_array(model.conv1.weight.data, 'conv1_weights.bin')
export_array(model.conv1.bias.data, 'conv1_bias.bin')
export_array(model.conv2.weight.data, 'conv2_weights.bin')
export_array(model.conv2.bias.data, 'conv2_bias.bin')
export_array(model.conv3.weight.data, 'conv3_weights.bin')
export_array(model.conv3.bias.data, 'conv3_bias.bin')

# Export FC weights
export_array(model.fc1.weight.data, 'fc1_weights.bin')
export_array(model.fc1.bias.data, 'fc1_bias.bin')
export_array(model.fc2.weight.data, 'fc2_weights.bin')
export_array(model.fc2.bias.data, 'fc2_bias.bin')
export_array(model.fc3.weight.data, 'fc3_weights.bin')
export_array(model.fc3.bias.data, 'fc3_bias.bin')

# Export BatchNorm
for i, bn in enumerate([model.bn1, model.bn2, model.bn3], 1):
    export_array(bn.weight.data, f'bn{i}_scale.bin')
    export_array(bn.bias.data, f'bn{i}_bias.bin')
    export_array(bn.running_mean.data, f'bn{i}_running_mean.bin')
    export_array(bn.running_var.data, f'bn{i}_running_var.bin')

print()

# Now reload using our loader
print("Reloading weights using model_loader...")
loaded_model = load_v28_model(export_dir, device=device)
print()

# Test accuracy after reload
print("Testing accuracy after reload...")
loaded_model.eval()
correct = 0

with torch.no_grad():
    for i in range(n_test):
        img = test_data[i].reshape((3, 32, 32))
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        output = loaded_model(img_tensor)
        pred = output.argmax(dim=1).item()
        if pred == test_labels[i]:
            correct += 1

accuracy_after = 100.0 * correct / n_test
print(f"  Accuracy after reload: {accuracy_after:.2f}%")
print()

# Compare
print("="*70)
print("Results")
print("="*70)
print(f"Accuracy before export: {accuracy_before:.2f}%")
print(f"Accuracy after reload:  {accuracy_after:.2f}%")
print()

if abs(accuracy_before - accuracy_after) < 0.1:
    print("✅ PERFECT! Export/load cycle works correctly!")
    print("   The export format is correct.")
else:
    print(f"❌ MISMATCH! Difference: {abs(accuracy_before - accuracy_after):.2f}%")
    print("   The export/load cycle is broken!")
