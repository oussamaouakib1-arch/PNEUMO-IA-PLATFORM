"""Send one real test radiograph to the local inference API."""

import base64
import csv
import json
from pathlib import Path

import requests

DATASET_ROOT = Path("data/raw/kaggle/chest_xray/chest_xray")
MANIFEST = Path("data/splits/manifest.csv")
OUTPUT_DIR = Path("artifacts/api_smoke")


def main() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        sample = next(
            row
            for row in csv.DictReader(stream)
            if row["split"] == "test" and row["label"] == "PNEUMONIA"
        )
    image_path = DATASET_ROOT / sample["relative_path"]
    with image_path.open("rb") as image:
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": (image_path.name, image, "image/jpeg")},
            timeout=60,
        )
    response.raise_for_status()
    result = response.json()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gradcam = result.pop("gradcam_base64")
    if gradcam:
        encoded = gradcam.split(",", maxsplit=1)[1]
        (OUTPUT_DIR / "gradcam.png").write_bytes(base64.b64decode(encoded))
    (OUTPUT_DIR / "response.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
