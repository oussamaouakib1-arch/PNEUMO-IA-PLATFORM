import torch

from pneumonia.training.model import BaselineCNN


def test_baseline_cnn_returns_two_logits_per_image() -> None:
    outputs = BaselineCNN()(torch.randn(2, 1, 64, 64))

    assert outputs.shape == (2, 2)
