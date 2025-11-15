"""
Oxford Flowers 102 Classification - PyTorch Implementation
Essential workflow for training and evaluating an image classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import numpy as np
from pathlib import Path
import time
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 102
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 5  # Early stopping patience

# Data directory
DATA_DIR = './data/oxford_flowers'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_data_transforms():
    """Define data transforms for training and validation/test

    Note: Using simple /255 normalization to match TensorFlow implementation,
    NOT ImageNet normalization. TF Hub's MobileNetV2 feature_vector expects [0,1] input.
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),  # Converts to [0, 1] range automatically
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to exactly 224x224 (matches TF)
        transforms.ToTensor(),  # Converts to [0, 1] range automatically
    ])

    return train_transforms, val_test_transforms


def load_datasets():
    """Load Oxford Flowers 102 dataset"""
    train_transforms, val_test_transforms = get_data_transforms()

    # Download and load datasets
    train_dataset = datasets.Flowers102(
        root=DATA_DIR,
        split='train',
        transform=train_transforms,
        download=True
    )

    val_dataset = datasets.Flowers102(
        root=DATA_DIR,
        split='val',
        transform=val_test_transforms,
        download=True
    )

    test_dataset = datasets.Flowers102(
        root=DATA_DIR,
        split='test',
        transform=val_test_transforms,
        download=True
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f'Dataset loaded:')
    print(f'  Training samples: {len(train_dataset)}')
    print(f'  Validation samples: {len(val_dataset)}')
    print(f'  Test samples: {len(test_dataset)}')

    return train_loader, val_loader, test_loader


def create_model(pretrained=True):
    """Create model with pretrained MobileNetV2 backbone

    IMPORTANT: Using SIMPLE architecture to match TensorFlow version:
    - TF model has only 130,662 trainable params (1280*102 + 102)
    - No hidden layers, just feature_extractor -> Dense(102)
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=pretrained)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Simple classifier to match TensorFlow (no hidden layers!)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),  # MobileNetV2 default
        nn.Linear(model.last_channel, NUM_CLASSES)  # 1280 -> 102
    )

    model = model.to(device)

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('\nModel architecture:')
    print(f'  Feature extractor: MobileNetV2 (frozen)')
    print(f'  Classifier: {model.last_channel} -> {NUM_CLASSES} (SIMPLE, matching TF)')
    print(f'  Trainable parameters: {trainable_params:,}')

    return model


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
    """Train the model with early stopping"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print('\nStarting training...')
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break

    elapsed_time = time.time() - start_time
    print(f'\nTraining completed in {elapsed_time/60:.2f} minutes')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best validation accuracy: {best_val_acc:.4f}')

    return history


def evaluate(model, test_loader):
    """Evaluate model on test set"""
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, criterion)

    print(f'\nTest Results:')
    print(f'  Loss: {test_loss:.4f}')
    print(f'  Accuracy: {test_acc:.4f}')

    return test_loss, test_acc


def process_image(image_path):
    """Process a PIL image for model input

    Matches TensorFlow preprocessing: resize to 224x224 and normalize to [0,1]
    """
    _, val_transforms = get_data_transforms()

    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor


def predict(image_path, model, class_names, top_k=5):
    """Predict top K classes for an image"""
    model.eval()

    image_tensor = process_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get top K predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Convert to class names if available
    if class_names:
        # Flowers102 labels are 0-indexed in PyTorch but 1-indexed in label_map.json
        top_classes = [class_names.get(str(idx + 1), f'Class {idx}') for idx in top_indices]
    else:
        top_classes = [str(idx) for idx in top_indices]

    return top_probs, top_classes


def load_label_map(filepath='label_map.json'):
    """Load class name mapping from JSON file"""
    try:
        with open(filepath, 'r') as f:
            class_names = json.load(f)
        print(f'Loaded {len(class_names)} class names from {filepath}')
        return class_names
    except FileNotFoundError:
        print(f'Warning: {filepath} not found. Using numeric labels.')
        return None


def main():
    """Main training and evaluation pipeline"""
    print('=' * 70)
    print('Oxford Flowers 102 Classification - PyTorch')
    print('=' * 70)

    # Load data
    train_loader, val_loader, test_loader = load_datasets()

    # Create model
    model = create_model(pretrained=True)

    # Train model
    history = train_model(model, train_loader, val_loader)

    # Load best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

    # Evaluate on test set
    evaluate(model, test_loader)

    # Save final model
    timestamp = int(time.time())
    final_model_path = f'oxford_flowers_model_{timestamp}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'\nFinal model saved to: {final_model_path}')

    # Example inference
    class_names = load_label_map()
    if class_names:
        print('\n' + '=' * 70)
        print('Example Inference')
        print('=' * 70)
        # Note: Update image_path to test with actual images
        # probs, classes = predict('./test_images/example.jpg', model, class_names)
        # for prob, cls in zip(probs, classes):
        #     print(f'{cls}: {prob:.4f}')

    print('\n✅ Training pipeline completed successfully!')


if __name__ == '__main__':
    main()
