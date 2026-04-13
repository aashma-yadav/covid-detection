import torch
import torch.nn as nn
import torchvision


class BasicCNN(nn.Module):
    """
    Simple 2-block CNN with BatchNorm and AdaptiveAvgPool2d.

    Changes from original:
      - Added BatchNorm2d after each conv for training stability
      - Replaced hardcoded flatten with AdaptiveAvgPool2d((4,4)) so the
        classifier head works at ANY input resolution (32×32, 224×224, etc.)
      - This fixes the RuntimeError: mat1/mat2 shape mismatch
    """

    def __init__(self, num_classes=3):
        super(BasicCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # AdaptiveAvgPool2d guarantees a fixed spatial output regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MiniVGG(nn.Module):
    """
    VGG-style CNN with double conv blocks and BatchNorm.

    Changes from original:
      - Added AdaptiveAvgPool2d((4,4)) to decouple spatial dims from classifier
      - Fixes the same shape crash as BasicCNN
    """

    def __init__(self, num_classes=3):
        super(MiniVGG, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # AdaptiveAvgPool2d guarantees a fixed spatial output regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class MiniResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MiniResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PretrainedResNet(nn.Module):
    """
    ResNet-18 with ImageNet-1k pretrained weights.

    Why this matters for medical imaging:
      - ImageNet pretraining provides rich low/mid-level feature extractors
        (edges, textures, shapes) that transfer well to X-ray analysis.
      - With only ~350 training images, training from scratch is highly prone
        to overfitting. Transfer learning is the SOTA approach here.
      - The backbone is frozen by default; only the classification head trains.
        Set freeze_backbone=False for fine-tuning after initial convergence.
    """

    def __init__(self, num_classes=3, freeze_backbone=True):
        super(PretrainedResNet, self).__init__()

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.base = torchvision.models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.base.parameters():
                param.requires_grad = False

        # Replace the final FC with a dropout + linear head
        in_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.base(x)