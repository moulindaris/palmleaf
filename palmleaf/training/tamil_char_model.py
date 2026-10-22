"""
training/tamil_char_model.py
CNN for Tamil character recognition — 71 classes.
Uses MobileNetV2 pretrained backbone (grayscale input adapted).
"""
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 71


class TamilCharCNN(nn.Module):
    """
    MobileNetV2 adapted for grayscale input, 71-class output.
    Pretrained on ImageNet — first conv layer replaced for 1-channel input.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base    = models.mobilenet_v2(weights=weights)

        # Replace first conv: 3-channel → 1-channel
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        # Initialize new conv by averaging pretrained RGB weights
        with torch.no_grad():
            base.features[0][0].weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

        self.features   = base.features
        in_features     = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_tamil_char_model(device: str = "cpu") -> TamilCharCNN:
    model = TamilCharCNN()
    model.to(device)
    return model
