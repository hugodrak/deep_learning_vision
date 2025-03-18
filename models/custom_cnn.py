"""
custom_cnn.py

Defines a simple custom CNN architecture for smaller datasets or demos.
"""

import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # for input images of size 224x224
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # (batch, 3, 224, 224) -> conv1 -> ReLU -> conv2 -> ReLU -> pool
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)  # shape: (batch, 64*56*56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
