"""Grad-CAM explanations for supported CNN architectures."""

from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as functional


def target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the final convolutional feature layer for a supported model."""
    if model_name == "baseline":
        baseline_features = cast(nn.Sequential, model.features)
        return cast(nn.Sequential, baseline_features[2])[0]
    if model_name == "resnet50":
        layer4 = cast(nn.Sequential, model.layer4)
        return cast(nn.Module, layer4[-1].conv3)
    if model_name == "densenet121":
        dense_features = cast(nn.Module, model.features)
        return cast(nn.Module, dense_features.norm5)
    if model_name == "efficientnet_b0":
        efficient_features = cast(nn.Sequential, model.features)
        return efficient_features[-1]
    raise ValueError(f"Unsupported model: {model_name}")


def compute_gradcam(
    model: nn.Module,
    inputs: Tensor,
    layer: nn.Module,
    class_index: int = 1,
) -> Tensor:
    """Return a normalized Grad-CAM heatmap for the first input image."""
    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        activations.append(output)

    def backward_hook(
        _module: nn.Module,
        _gradient_input: tuple[Tensor | None, ...],
        gradient_output: tuple[Tensor | None, ...],
    ) -> None:
        if gradient_output[0] is not None:
            gradients.append(gradient_output[0])

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        logits[:, class_index].sum().backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(
            heatmap,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        maximum = heatmap.max()
        if maximum > 0:
            heatmap = heatmap / maximum
        return heatmap.detach().cpu()
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_overlay(
    original_path: Path,
    heatmap: Tensor,
    output_path: Path,
    opacity: float = 0.45,
) -> None:
    """Overlay a red-yellow heatmap on the original radiograph."""
    with Image.open(original_path) as image:
        base = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    heat = np.asarray(heatmap)
    color = np.zeros((*heat.shape, 3), dtype=np.uint8)
    color[..., 0] = np.clip(heat * 255, 0, 255).astype(np.uint8)
    color[..., 1] = np.clip(np.maximum(0, (heat - 0.5) * 2) * 255, 0, 255).astype(np.uint8)
    overlay = Image.blend(base, Image.fromarray(color), opacity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
