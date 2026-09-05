import io

import pytest
from PIL import Image

from pneumonia.api.inference import InvalidImageError, PredictionService


def test_open_image_accepts_valid_grayscale_image() -> None:
    buffer = io.BytesIO()
    Image.new("L", (64, 64), color=100).save(buffer, format="PNG")

    image = PredictionService._open_image(buffer.getvalue())

    assert image.mode == "L"
    assert image.size == (64, 64)


def test_open_image_rejects_non_image_bytes() -> None:
    with pytest.raises(InvalidImageError, match="image valide"):
        PredictionService._open_image(b"not-an-image")


def test_open_image_rejects_tiny_image() -> None:
    buffer = io.BytesIO()
    Image.new("L", (16, 16), color=100).save(buffer, format="PNG")

    with pytest.raises(InvalidImageError, match="32 x 32"):
        PredictionService._open_image(buffer.getvalue())
