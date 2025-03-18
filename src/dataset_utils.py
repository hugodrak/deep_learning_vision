"""
dataset_utils.py

Provides functionality for loading image datasets with transformations and augmentations.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Creates PyTorch DataLoader objects for train and validation sets.

    Args:
        data_dir (str): Path to the data folder containing 'train' and 'val' subfolders.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        (DataLoader, DataLoader) for train_loader and val_loader.
    """

    # ImageNet-like statistics for normalization (if needed)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader
