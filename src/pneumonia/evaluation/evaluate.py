"""Evaluate a checkpoint, tune its threshold, and export prediction details."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pneumonia.evaluation.calibration import calibration_metrics, select_threshold
from pneumonia.evaluation.gradcam import compute_gradcam, save_overlay, target_layer
from pneumonia.training.dataset import ChestXrayDataset
from pneumonia.training.factory import ModelName, create_model
from pneumonia.training.metrics import binary_metrics
from pneumonia.training.transforms import build_transforms


def load_checkpoint(checkpoint_path: Path) -> tuple[nn.Module, dict[str, Any], ModelName]:
    """Restore a supported model and its training configuration."""
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    model_name: ModelName = config.get("model_name", "baseline")
    model = create_model(
        model_name,
        pretrained=False,
        freeze_backbone=False,
        dropout=float(config.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, model_name


def collect_predictions(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
) -> tuple[list[int], list[float]]:
    """Collect labels and pneumonia probabilities."""
    targets: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            probabilities.extend(torch.softmax(model(images), dim=1)[:, 1].tolist())
            targets.extend(labels.tolist())
    return targets, probabilities


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    minimum_recall: float = 0.90,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Tune on validation data, evaluate test data, and create a Grad-CAM."""
    model, config, model_name = load_checkpoint(checkpoint_path)
    image_size = int(config.get("image_size", 224))
    channels = 1 if model_name == "baseline" else 3
    _, transform = build_transforms(image_size, channels)
    validation_dataset = ChestXrayDataset(manifest_path, dataset_root, "val", transform)
    test_dataset = ChestXrayDataset(manifest_path, dataset_root, "test", transform)
    validation_targets, validation_probabilities = collect_predictions(
        model,
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False),
    )
    threshold_result = select_threshold(
        validation_targets, validation_probabilities, minimum_recall
    )
    threshold = float(threshold_result["threshold"])
    test_targets, test_probabilities = collect_predictions(
        model,
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
    test_predictions = [int(probability >= threshold) for probability in test_probabilities]
    result = {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "threshold_selection": threshold_result,
        "validation_calibration": calibration_metrics(validation_targets, validation_probabilities),
        "test_calibration": calibration_metrics(test_targets, test_probabilities),
        "test_metrics": binary_metrics(test_targets, test_predictions, test_probabilities),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["relative_path", "target", "probability", "prediction"],
        )
        writer.writeheader()
        for sample, target, probability, prediction in zip(
            test_dataset.samples,
            test_targets,
            test_probabilities,
            test_predictions,
            strict=True,
        ):
            writer.writerow(
                {
                    "relative_path": sample["relative_path"],
                    "target": target,
                    "probability": probability,
                    "prediction": prediction,
                }
            )

    pneumonia_index = next(
        index for index, sample in enumerate(test_dataset.samples) if sample["label"] == "PNEUMONIA"
    )
    tensor, _ = test_dataset[pneumonia_index]
    heatmap = compute_gradcam(model, tensor.unsqueeze(0), target_layer(model, model_name))
    sample_path = dataset_root / test_dataset.samples[pneumonia_index]["relative_path"]
    save_overlay(sample_path, heatmap, output_dir / "gradcam_pneumonia.png")
    return result


def main() -> int:
    """Run checkpoint evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate and explain a pneumonia model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        args.dataset_root,
        args.output_dir,
        args.minimum_recall,
        args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
