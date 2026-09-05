import pytest
from torch import nn

from pneumonia.training.factory import (
    MODEL_NAMES,
    create_model,
    total_parameter_count,
    trainable_parameter_count,
)
from pneumonia.training.transforms import build_transforms


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_factory_creates_binary_classifier(model_name: str) -> None:
    model = create_model(model_name, pretrained=False, freeze_backbone=model_name != "baseline")

    assert isinstance(model, nn.Module)
    assert trainable_parameter_count(model) > 0
    assert total_parameter_count(model) >= trainable_parameter_count(model)
    if model_name != "baseline":
        assert trainable_parameter_count(model) < total_parameter_count(model)


def test_rgb_transforms_produce_three_channels() -> None:
    from PIL import Image

    _, evaluation = build_transforms(image_size=32, channels=3)
    tensor = evaluation(Image.new("L", (48, 48), color=100))

    assert tensor.shape == (3, 32, 32)


def test_invalid_channel_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="channels"):
        build_transforms(channels=2)
