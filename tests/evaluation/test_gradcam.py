from pathlib import Path

import torch
from PIL import Image

from pneumonia.evaluation.gradcam import compute_gradcam, save_overlay, target_layer
from pneumonia.training.model import BaselineCNN


def test_gradcam_is_normalized_and_overlay_is_written(tmp_path: Path) -> None:
    model = BaselineCNN()
    inputs = torch.randn(1, 1, 32, 32)

    heatmap = compute_gradcam(model, inputs, target_layer(model, "baseline"))

    assert heatmap.shape == (32, 32)
    assert float(heatmap.min()) >= 0
    assert float(heatmap.max()) <= 1
    original = tmp_path / "xray.jpeg"
    output = tmp_path / "overlay.png"
    Image.new("L", (32, 32), color=100).save(original)
    save_overlay(original, heatmap, output)
    assert output.exists()
