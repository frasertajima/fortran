"""
Standalone inference script for Oxford Flowers 102 classifier
Usage: python inference.py --image path/to/image.jpg --model best_model.pth
"""

import torch
import argparse
from torchvision import transforms, models
from PIL import Image
import json


def load_model(model_path, num_classes=102):
    """Load trained model from checkpoint

    Note: Using SIMPLE architecture matching TensorFlow version
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model architecture (SIMPLE version matching TF)
    model = models.mobilenet_v2(pretrained=False)

    # Simple classifier (no hidden layers)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model.last_channel, num_classes)  # 1280 -> 102
    )

    # Load weights
    if model_path.endswith('.pth') and 'model_state_dict' in torch.load(model_path, map_location=device):
        # Checkpoint format
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    return model, device


def process_image(image_path):
    """Process image for model input

    Note: Using simple /255 normalization (via ToTensor) to match TensorFlow,
    NOT ImageNet normalization. Matches TF's image /= 255.0
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to exactly 224x224 (matches TF)
        transforms.ToTensor(),  # Converts to [0, 1] automatically
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict(image_path, model, device, class_names=None, top_k=5):
    """Predict top K classes for an image"""
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get top K predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Convert to class names
    if class_names:
        # PyTorch Flowers102 uses 0-indexing, label_map.json uses 1-indexing
        top_classes = [class_names.get(str(idx + 1), f'Class {idx}') for idx in top_indices]
    else:
        top_classes = [str(idx) for idx in top_indices]

    return top_probs, top_classes


def load_label_map(filepath='label_map.json'):
    """Load class name mapping"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'Warning: {filepath} not found. Using numeric labels.')
        return None


def main():
    parser = argparse.ArgumentParser(description='Oxford Flowers 102 Classifier - Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--labels', type=str, default='label_map.json', help='Path to label mapping JSON')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to show')

    args = parser.parse_args()

    print('Loading model...')
    model, device = load_model(args.model)
    print(f'Model loaded successfully on {device}')

    print('Loading class labels...')
    class_names = load_label_map(args.labels)

    print(f'Processing image: {args.image}')
    probs, classes = predict(args.image, model, device, class_names, args.top_k)

    print(f'\nTop {args.top_k} Predictions:')
    print('-' * 50)
    for i, (prob, cls) in enumerate(zip(probs, classes), 1):
        print(f'{i}. {cls:30s} {prob:.4f} ({prob*100:.2f}%)')


if __name__ == '__main__':
    main()
