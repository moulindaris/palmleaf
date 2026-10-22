"""
training/classification_model.py
ResNet-18 fine-tuned for Tamil manuscript category classification.
Output: 6 classes — Medicine, Astrology, Literature, Philosophy, Mathematics, History
"""
import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 6


def get_classifier(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Load ResNet-18 with pretrained ImageNet weights.
    Replace final FC layer for our num_classes.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    # Freeze early layers — only train last 2 blocks + FC
    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def get_mobilenet(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Lighter alternative: MobileNetV2 — good for small datasets.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model   = models.mobilenet_v2(weights=weights)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model
