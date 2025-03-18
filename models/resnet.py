"""
resnet.py

Defines a ResNet-based classifier by leveraging PyTorch's torchvision models.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        """
        Initialize a ResNet-based model for classification.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet-pretrained weights.
        """
        super(ResNetClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=pretrained)
        # Replace the final fully connected layer to match num_classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
