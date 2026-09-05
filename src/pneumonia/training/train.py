"""Reproducible baseline CNN training with optional MLflow tracking."""

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset

from pneumonia.config import get_settings
from pneumonia.logging import configure_logging
from pneumonia.training.dataset import ChestXrayDataset
from pneumonia.training.factory import MODEL_NAMES, ModelName, create_model
from pneumonia.training.metrics import binary_metrics
from pneumonia.training.transforms import build_transforms

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a reproducible baseline run."""

    dataset_root: Path
    manifest_path: Path = Path("data/splits/manifest.csv")
    output_dir: Path = Path("models/baseline")
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    dropout: float = 0.3
    num_workers: int = 0
    patience: int = 3
    seed: int = 42
    max_batches: int | None = None
    use_mlflow: bool = True
    model_name: ModelName = "baseline"
    pretrained: bool = False
    freeze_backbone: bool = False


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(dataset: ChestXrayDataset, device: torch.device) -> Tensor:
    """Calculate inverse-frequency class weights."""
    counts = dataset.label_counts()
    total = sum(counts.values())
    return torch.tensor(
        [total / (2 * counts[index]) for index in (0, 1)],
        dtype=torch.float32,
        device=device,
    )


def balanced_subset(
    dataset: ChestXrayDataset,
    max_samples: int,
    seed: int,
) -> Subset[tuple[Tensor, int]]:
    """Select an approximately class-balanced subset without replacement."""
    indices_by_label: dict[str, list[int]] = {"NORMAL": [], "PNEUMONIA": []}
    for index, sample in enumerate(dataset.samples):
        indices_by_label[sample["label"]].append(index)
    rng = random.Random(seed)
    for indices in indices_by_label.values():
        rng.shuffle(indices)
    per_class = max_samples // 2
    selected = indices_by_label["NORMAL"][:per_class] + indices_by_label["PNEUMONIA"][:per_class]
    if len(selected) < max_samples:
        remaining = [
            index for indices in indices_by_label.values() for index in indices[per_class:]
        ]
        selected.extend(remaining[: max_samples - len(selected)])
    return Subset(dataset, sorted(selected))


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist())
    if not losses:
        raise RuntimeError("No batches were processed.")
    return float(np.mean(losses)), binary_metrics(targets, predictions, probabilities)


def train_baseline(config: TrainingConfig) -> dict[str, Any]:
    """Train, select by validation recall, and evaluate the baseline CNN."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 1 if config.model_name == "baseline" else 3
    train_transform, eval_transform = build_transforms(config.image_size, channels)
    datasets = {
        split: ChestXrayDataset(
            config.manifest_path,
            config.dataset_root,
            split,
            train_transform if split == "train" else eval_transform,
        )
        for split in ("train", "val", "test")
    }
    loader_datasets: dict[str, Dataset[tuple[Tensor, int]]] = dict(datasets)
    if config.max_batches is not None:
        max_samples = config.max_batches * config.batch_size
        loader_datasets = {
            split: balanced_subset(dataset, max_samples, config.seed + index)
            for index, (split, dataset) in enumerate(datasets.items())
        }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            loader_datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "val": DataLoader(
            loader_datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            loader_datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        ),
    }
    model = create_model(
        config.model_name,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(datasets["train"], device))
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_recall = -1.0
    stale_epochs = 0

    if config.use_mlflow:
        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        mlflow.set_experiment("pneumonia-model-comparison")
        mlflow.start_run(run_name=config.model_name)
        mlflow.log_params(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            }
        )
    try:
        for epoch in range(1, config.epochs + 1):
            train_loss, train_metrics = run_epoch(
                model,
                loaders["train"],
                criterion,
                device,
                optimizer,
                config.max_batches,
            )
            val_loss, val_metrics = run_epoch(
                model, loaders["val"], criterion, device, max_batches=config.max_batches
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            LOGGER.info("Epoch %d validation recall: %.4f", epoch, val_metrics["recall"])
            if config.use_mlflow:
                mlflow.log_metrics(
                    {key: float(value) for key, value in row.items() if key != "epoch"},
                    step=epoch,
                )
            recall = float(val_metrics["recall"])
            if recall > best_recall:
                best_recall, stale_epochs = recall, 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "validation_metrics": val_metrics,
                        "config": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in asdict(config).items()
                        },
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_metrics = run_epoch(
            model, loaders["test"], criterion, device, max_batches=config.max_batches
        )
        result = {
            "device": str(device),
            "best_validation_recall": best_recall,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "history": history,
            "checkpoint": str(checkpoint_path),
        }
        metrics_path = config.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if config.use_mlflow:
            mlflow.log_metrics({f"test_{key}": float(value) for key, value in test_metrics.items()})
            mlflow.log_artifacts(str(config.output_dir))
        return result
    finally:
        if config.use_mlflow:
            mlflow.end_run()


def main() -> int:
    """Train from command-line arguments."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    configure_logging()
    parser = argparse.ArgumentParser(description="Train the baseline pneumonia CNN.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--model", choices=MODEL_NAMES, default="baseline")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("models") / args.model
    train_baseline(
        TrainingConfig(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            max_batches=args.max_batches,
            use_mlflow=not args.no_mlflow,
            model_name=args.model,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
