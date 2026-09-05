"""Model factory for the baseline and transfer-learning architectures."""

from typing import Literal, cast

from torch import nn
from torchvision import models

from pneumonia.training.model import BaselineCNN

ModelName = Literal["baseline", "resnet50", "densenet121", "efficientnet_b0"]
MODEL_NAMES: tuple[ModelName, ...] = (
    "baseline",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
)


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def create_model(
    name: ModelName,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Create a binary classifier with an optionally frozen pretrained backbone."""
    if name == "baseline":
        return BaselineCNN(dropout)

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            _freeze(model)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 2))
        return cast(nn.Module, model)

    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        if freeze_backbone:
            _freeze(model)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier.in_features, 2),
        )
        return cast(nn.Module, model)

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            _freeze(model)
        input_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_features, 2))
        return cast(nn.Module, model)

    raise ValueError(f"Unsupported model: {name}")


def trainable_parameter_count(model: nn.Module) -> int:
    """Count parameters that the optimizer will update."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    """Count all model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())
