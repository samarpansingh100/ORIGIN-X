"""
Artificial Neural Network (ANN) and Transfer Learning architectures for ORIGIN X.
Designed to learn spatial, spectral, and micro-texture representations
while preventing shortcut learning on surface color biases.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from originx.config import MODELS_DIR, IMAGE_TARGET_SIZE


class OriginX_ConvNet(nn.Module):
    """
    Multi-scale convolutional neural network tailored for forensic artifact detection.
    Features parallel convolutional filters to capture both micro-texture sensor noise (PRNU)
    and broader spatial frequency coherence without relying on single-filter shortcuts.
    """
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        # Stream 1: Fine-grained 3x3 receptive field for noise/checkerboard artifacts
        self.conv1_fine = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # Stream 2: Wider 5x5 receptive field for spatial coherence
        self.conv1_wide = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_fine = F.leaky_relu(self.conv1_fine(x), 0.1)
        f_wide = F.leaky_relu(self.conv1_wide(x), 0.1)
        x = torch.cat([f_fine, f_wide], dim=1)
        x = self.pool1(self.bn1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.classifier(x)
        return logits


class OriginX_TransferNet(nn.Module):
    """
    Transfer learning network utilizing a lightweight ResNet-18 backbone
    with adapted forensic head and temperature calibration support.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # Extract convolutional stages
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        in_features = backbone.fc.in_features  # 512
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits
