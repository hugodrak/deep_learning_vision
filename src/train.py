"""
train.py

Training script for image classification using PyTorch.
Choose between a CustomCNN or a ResNet-based model.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.custom_cnn import CustomCNN
from models.resnet import ResNetClassifier
from dataset_utils import create_dataloaders

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
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

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to the dataset root (contains train/ and val/).")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "custom_cnn"],
                        help="Which model to use.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    # Create model
    if args.model == "resnet":
        model = ResNetClassifier(num_classes=args.num_classes, pretrained=True)
    else:
        model = CustomCNN(num_classes=args.num_classes)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.model}_best.pth")
            print(f"New best model saved with Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
