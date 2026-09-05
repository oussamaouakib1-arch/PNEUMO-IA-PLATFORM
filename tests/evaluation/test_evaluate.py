import csv
from pathlib import Path

import torch
from PIL import Image

from pneumonia.evaluation.evaluate import evaluate_checkpoint
from pneumonia.training.model import BaselineCNN


def _create_evaluation_data(root: Path, manifest: Path) -> None:
    rows: list[dict[str, str]] = []
    for split in ("val", "test"):
        for label, value in (("NORMAL", 40), ("PNEUMONIA", 180)):
            relative_path = f"{split}/{label}/{split}-{label}.jpeg"
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (40, 40), color=value).save(path)
            rows.append(
                {
                    "relative_path": relative_path,
                    "label": label,
                    "patient_id": f"{split}-{label}",
                    "split": split,
                    "sha256": f"{split}-{label}",
                }
            )
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_checkpoint_writes_report_predictions_and_gradcam(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    manifest = tmp_path / "manifest.csv"
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "evaluation"
    _create_evaluation_data(dataset_root, manifest)
    model = BaselineCNN()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"model_name": "baseline", "image_size": 32, "dropout": 0.3},
        },
        checkpoint,
    )

    result = evaluate_checkpoint(
        checkpoint,
        manifest,
        dataset_root,
        output,
        minimum_recall=0.5,
        batch_size=2,
    )

    assert result["model"] == "baseline"
    assert (output / "evaluation.json").exists()
    assert (output / "predictions.csv").exists()
    assert (output / "gradcam_pneumonia.png").exists()
