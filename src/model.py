"""
Model architectures for Diabetic Retinopathy Grading in Federated Learning:
1. RetinalNet: ResNet-based backbone with GroupNorm (transfer learning with ImageNet weights, FedBN & DP-SGD compliant).
2. SmallCNN: 5-block convolutional architecture with GroupNorm.
Both use GroupNorm (per-sample statistics) instead of BatchNorm, enabling:
- FedBN personalization (GroupNorm affine parameters remain strictly local to each hospital).
- DP-SGD compatibility (per-sample gradient clipping without batch contamination).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional


def _convert_bn_to_gn(module: nn.Module, num_groups: int = 8):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm layers,
    transferring the affine parameters (gamma, beta) to preserve pretrained representations.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = min(num_groups, c)
            while c % g != 0 and g > 1:
                g -= 1
            gn = nn.GroupNorm(num_groups=g, num_channels=c, affine=True)
            with torch.no_grad():
                gn.weight.copy_(child.weight)
                gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            _convert_bn_to_gn(child, num_groups)


class RetinalNet(nn.Module):
    """
    Pretrained ResNet18 adapted with GroupNorm for FedBN and DP-SGD.
    Provides transfer-learned feature representations for Diabetic Retinopathy lesions:
    microaneurysms, hemorrhages, hard exudates, cotton wool spots, and neovascularization.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, num_groups: int = 8, dropout: float = 0.25):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        # Convert BatchNorm to GroupNorm for FedBN and DP-SGD
        _convert_bn_to_gn(base, num_groups=num_groups)

        self.conv1 = base.conv1
        self.bn1 = base.bn1  # Converted to GroupNorm
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # Target layer for Grad-CAM

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(512, 128)
        self.act_fc = nn.LeakyReLU(0.2, inplace=True)
        self.classifier = nn.Linear(128, num_classes)

        # Hook storage for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad: torch.Tensor):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Layer 4 is the target for Grad-CAM
        x = self.layer4(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act_fc(x)
        logits = self.classifier(x)
        return logits

    def get_target_layer_activations(self) -> torch.Tensor:
        return self.activations

    def get_target_layer_gradients(self) -> torch.Tensor:
        return self.gradients

    def get_shared_param_names(self) -> List[str]:
        shared = []
        for name, _ in self.named_parameters():
            if not self.is_local_param(name):
                shared.append(name)
        return shared

    def get_local_param_names(self) -> List[str]:
        local = []
        for name, _ in self.named_parameters():
            if self.is_local_param(name):
                local.append(name)
        return local

    @staticmethod
    def is_local_param(param_name: str) -> bool:
        """GroupNorm parameters are local in FedBN."""
        return "gn.weight" in param_name or "gn.bias" in param_name or "bn1.weight" in param_name or "bn1.bias" in param_name or ".bn" in param_name or ".downsample.1" in param_name


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class SmallCNN(nn.Module):
    """5-block Convolutional Neural Network with GroupNorm for Diabetic Retinopathy."""
    def __init__(self, in_channels: int = 3, num_classes: int = 5, num_groups: int = 8, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.block1 = ConvBlock(in_channels, 32, num_groups=min(8, 32), pool=True)
        self.block2 = ConvBlock(32, 64, num_groups=min(8, 64), pool=True)
        self.block3 = ConvBlock(64, 128, num_groups=min(8, 128), pool=True)
        self.block4 = ConvBlock(128, 256, num_groups=min(8, 256), pool=True)
        self.block5 = ConvBlock(256, 256, num_groups=min(8, 256), pool=False)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256, 128)
        self.act_fc = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Linear(128, num_classes)

        self.gradients = None
        self.activations = None

    def activations_hook(self, grad: torch.Tensor):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.block5.conv(x)
        x = self.block5.gn(x)
        x = self.block5.act(x)
        
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act_fc(x)
        logits = self.classifier(x)
        return logits

    def get_target_layer_activations(self) -> torch.Tensor:
        return self.activations

    def get_target_layer_gradients(self) -> torch.Tensor:
        return self.gradients

    def get_shared_param_names(self) -> List[str]:
        shared = []
        for name, _ in self.named_parameters():
            if not self.is_local_param(name):
                shared.append(name)
        return shared

    def get_local_param_names(self) -> List[str]:
        local = []
        for name, _ in self.named_parameters():
            if self.is_local_param(name):
                local.append(name)
        return local

    @staticmethod
    def is_local_param(param_name: str) -> bool:
        return "gn.weight" in param_name or "gn.bias" in param_name


def get_model(
    model_name: str = "RetinalNet",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.25
) -> nn.Module:
    """Factory function for model creation."""
    if model_name.lower() in ["retinalnet", "resnet", "resnet18"]:
        return RetinalNet(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        return SmallCNN(num_classes=num_classes, dropout=dropout)
