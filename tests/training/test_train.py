import csv
from pathlib import Path

from PIL import Image

from pneumonia.training.train import TrainingConfig, train_baseline


def _build_tiny_dataset(root: Path, manifest: Path) -> None:
    rows: list[dict[str, str]] = []
    for split_index, split in enumerate(("train", "val", "test")):
        for label_index, label in enumerate(("NORMAL", "PNEUMONIA")):
            relative_path = f"{split}/{label}/patient-{split}-{label}.jpeg"
            image_path = root / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "L",
                size=(48, 48),
                color=40 + split_index * 30 + label_index * 100,
            ).save(image_path)
            rows.append(
                {
                    "relative_path": relative_path,
                    "label": label,
                    "patient_id": f"patient-{split}-{label}",
                    "split": split,
                    "sha256": f"hash-{split}-{label}",
                }
            )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_train_baseline_creates_checkpoint_and_metrics(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "model"
    _build_tiny_dataset(dataset_root, manifest)

    result = train_baseline(
        TrainingConfig(
            dataset_root=dataset_root,
            manifest_path=manifest,
            output_dir=output_dir,
            image_size=32,
            batch_size=2,
            epochs=1,
            max_batches=1,
            use_mlflow=False,
        )
    )

    assert result["device"] == "cpu"
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert "recall" in result["test_metrics"]
