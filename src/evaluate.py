"""
evaluate.py

Loads a trained model and evaluates it on the validation set, reporting accuracy.
"""

import argparse
import torch
import torch.nn as nn
from models.custom_cnn import CustomCNN
from models.resnet import ResNetClassifier
from dataset_utils import create_dataloaders

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / total
    acc = 100.0 * correct / total
    return loss, acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained image classification model.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to the dataset root (contains train/ and val/).")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "custom_cnn"],
                        help="Which model to load.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes.")
    parser.add_argument("--checkpoint", type=str, default="resnet_best.pth",
                        help="Path to the model checkpoint file.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create val dataloader
    _, val_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    # Load model
    if args.model == "resnet":
        model = ResNetClassifier(num_classes=args.num_classes, pretrained=False)
    else:
        model = CustomCNN(num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
