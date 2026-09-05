"""Checkpoint-backed inference service."""

import base64
import io
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from pneumonia.evaluation.evaluate import load_checkpoint
from pneumonia.evaluation.gradcam import compute_gradcam, target_layer
from pneumonia.training.transforms import build_transforms

MEDICAL_WARNING = (
    "Prototype académique d'aide à la décision. Ce résultat ne remplace pas l'avis d'un radiologue."
)


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported radiograph."""


class PredictionService:
    """Load one model and serve deterministic single-image predictions."""

    def __init__(
        self,
        checkpoint_path: Path = Path("models/baseline/best_model.pt"),
        evaluation_path: Path = Path("artifacts/evaluation/baseline/evaluation.json"),
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.model, self.config, self.model_name = load_checkpoint(checkpoint_path)
        self.threshold = self._load_threshold(evaluation_path)
        image_size = int(self.config.get("image_size", 224))
        channels = 1 if self.model_name == "baseline" else 3
        _, self.transform = build_transforms(image_size, channels)
        self.layer = target_layer(self.model, self.model_name)

    @staticmethod
    def _load_threshold(evaluation_path: Path) -> float:
        if not evaluation_path.exists():
            return 0.5
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        return float(payload["threshold_selection"]["threshold"])

    @staticmethod
    def _open_image(content: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(content)) as source:
                source.verify()
            with Image.open(io.BytesIO(content)) as source:
                image: Image.Image = source.convert("L")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("Le fichier ne contient pas une image valide.") from exc
        if image.width < 32 or image.height < 32:
            raise InvalidImageError("L'image doit mesurer au moins 32 x 32 pixels.")
        return image

    @staticmethod
    def _gradcam_data_url(image: Image.Image, heatmap: torch.Tensor) -> str:
        heat = np.asarray(heatmap)
        base = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
        color = np.zeros((*heat.shape, 3), dtype=np.uint8)
        color[..., 0] = np.clip(heat * 255, 0, 255).astype(np.uint8)
        color[..., 1] = np.clip(np.maximum(0, (heat - 0.5) * 2) * 255, 0, 255).astype(np.uint8)
        overlay = Image.blend(base, Image.fromarray(color), 0.45)
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict(self, content: bytes, include_gradcam: bool = True) -> dict[str, Any]:
        """Predict one upload and optionally return an explanatory heatmap."""
        image = self._open_image(content)
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            probability = float(torch.softmax(self.model(tensor), dim=1)[0, 1])
        label = "PNEUMONIA" if probability >= self.threshold else "NORMAL"
        gradcam = None
        if include_gradcam:
            heatmap = compute_gradcam(self.model, tensor, self.layer)
            gradcam = self._gradcam_data_url(image, heatmap)
        return {
            "request_id": str(uuid4()),
            "prediction": label,
            "pneumonia_probability": probability,
            "normal_probability": 1.0 - probability,
            "threshold": self.threshold,
            "model": self.model_name,
            "gradcam_base64": gradcam,
            "warning": MEDICAL_WARNING,
        }
