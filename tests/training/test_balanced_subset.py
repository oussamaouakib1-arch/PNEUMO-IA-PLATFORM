import csv
from pathlib import Path

import torch

from pneumonia.training.dataset import ChestXrayDataset
from pneumonia.training.train import balanced_subset


def test_balanced_subset_contains_both_classes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "relative_path": f"image-{index}.jpeg",
            "label": "NORMAL" if index < 8 else "PNEUMONIA",
            "patient_id": f"patient-{index}",
            "split": "train",
            "sha256": f"hash-{index}",
        }
        for index in range(16)
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset = ChestXrayDataset(manifest, tmp_path, "train", lambda image: torch.zeros(1))

    subset = balanced_subset(dataset, max_samples=6, seed=42)
    labels = [dataset.samples[index]["label"] for index in subset.indices]

    assert labels.count("NORMAL") == 3
    assert labels.count("PNEUMONIA") == 3
