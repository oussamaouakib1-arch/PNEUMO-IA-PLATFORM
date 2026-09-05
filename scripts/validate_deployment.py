"""Validate deployed API, UI and one real prediction."""

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def validate_deployment(api_url: str, ui_url: str, image_path: Path) -> dict[str, Any]:
    """Run deployment health checks and a real inference request."""
    health_response = requests.get(f"{api_url}/health", timeout=30)
    health_response.raise_for_status()

    ui_response = requests.get(f"{ui_url}/_stcore/health", timeout=30)
    ui_response.raise_for_status()

    with image_path.open("rb") as image_file:
        prediction_response = requests.post(
            f"{api_url}/predict",
            params={"include_gradcam": "false"},
            files={"file": (image_path.name, image_file, "image/jpeg")},
            timeout=120,
        )
    prediction_response.raise_for_status()
    prediction = prediction_response.json()

    metrics_response = requests.get(f"{api_url}/metrics", timeout=30)
    metrics_response.raise_for_status()
    if "pneumonia_predictions_total" not in metrics_response.text:
        raise RuntimeError("Prediction metrics are missing.")

    return {
        "api": health_response.json(),
        "ui_status": ui_response.text.strip(),
        "prediction": {
            "label": prediction["prediction"],
            "pneumonia_probability": prediction["pneumonia_probability"],
            "request_id": prediction["request_id"],
        },
        "metrics": "available",
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ui-url", default="http://127.0.0.1:8501")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    result = validate_deployment(args.api_url.rstrip("/"), args.ui_url.rstrip("/"), args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
