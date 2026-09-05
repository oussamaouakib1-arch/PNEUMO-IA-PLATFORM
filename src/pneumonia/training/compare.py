"""Train multiple architectures under a shared experimental protocol."""

import argparse
import json
from pathlib import Path
from typing import Any

from pneumonia.logging import configure_logging
from pneumonia.training.factory import MODEL_NAMES, ModelName
from pneumonia.training.train import TrainingConfig, train_baseline


def compare_models(
    *,
    dataset_root: Path,
    manifest_path: Path,
    model_names: list[ModelName],
    output_root: Path,
    epochs: int,
    batch_size: int,
    image_size: int,
    max_batches: int | None,
    pretrained: bool,
    freeze_backbone: bool,
    use_mlflow: bool,
) -> list[dict[str, Any]]:
    """Train requested models and write a metric-sorted comparison report."""
    results: list[dict[str, Any]] = []
    for model_name in model_names:
        result = train_baseline(
            TrainingConfig(
                dataset_root=dataset_root,
                manifest_path=manifest_path,
                output_dir=output_root / model_name,
                model_name=model_name,
                pretrained=pretrained and model_name != "baseline",
                freeze_backbone=freeze_backbone and model_name != "baseline",
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                max_batches=max_batches,
                use_mlflow=use_mlflow,
            )
        )
        metrics = result["test_metrics"]
        results.append(
            {
                "model": model_name,
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "false_negative": metrics["false_negative"],
                "checkpoint": result["checkpoint"],
            }
        )
    results.sort(key=lambda row: (row["recall"], row["f1"]), reverse=True)
    report_path = output_root / "comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> int:
    """Run a model comparison from the command line."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Compare pneumonia CNN architectures.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.csv"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output-root", type=Path, default=Path("models/comparison"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    compare_models(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        model_names=args.models,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_batches=args.max_batches,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze,
        use_mlflow=not args.no_mlflow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
