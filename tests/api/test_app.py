from typing import Any

from fastapi.testclient import TestClient

from pneumonia.api.app import create_app
from pneumonia.api.inference import InvalidImageError, PredictionService


class FakePredictionService(PredictionService):
    def __init__(self) -> None:
        self.model_name = "baseline"
        self.threshold = 0.67

    def predict(self, content: bytes, include_gradcam: bool = True) -> dict[str, Any]:
        if content == b"invalid":
            raise InvalidImageError("Image invalide")
        return {
            "request_id": "request-1",
            "prediction": "PNEUMONIA",
            "pneumonia_probability": 0.82,
            "normal_probability": 0.18,
            "threshold": self.threshold,
            "model": self.model_name,
            "gradcam_base64": "data:image/png;base64,AA==" if include_gradcam else None,
            "warning": "Prototype académique",
        }


def test_health_reports_loaded_model() -> None:
    with TestClient(create_app(FakePredictionService())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model": "baseline",
        "threshold": 0.67,
    }


def test_predict_returns_validated_response() -> None:
    with TestClient(create_app(FakePredictionService())) as client:
        response = client.post(
            "/predict?include_gradcam=false",
            files={"file": ("xray.jpeg", b"image", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["prediction"] == "PNEUMONIA"
    assert response.json()["gradcam_base64"] is None


def test_predict_rejects_unsupported_content_type() -> None:
    with TestClient(create_app(FakePredictionService())) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"text", "text/plain")},
        )

    assert response.status_code == 415


def test_predict_maps_invalid_image_to_422() -> None:
    with TestClient(create_app(FakePredictionService())) as client:
        response = client.post(
            "/predict",
            files={"file": ("xray.png", b"invalid", "image/png")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Image invalide"


def test_metrics_expose_http_and_prediction_counters() -> None:
    with TestClient(create_app(FakePredictionService())) as client:
        prediction = client.post(
            "/predict?include_gradcam=false",
            files={"file": ("xray.jpeg", b"image", "image/jpeg")},
        )
        response = client.get("/metrics")

    assert prediction.status_code == 200
    assert response.status_code == 200
    assert "pneumonia_http_requests_total" in response.text
    assert 'path="/predict"' in response.text
    assert "pneumonia_predictions_total" in response.text
    assert 'prediction="PNEUMONIA"' in response.text
