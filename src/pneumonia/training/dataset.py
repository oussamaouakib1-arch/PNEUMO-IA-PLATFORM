"""PyTorch dataset backed by the leakage-safe phase 3 manifest."""

import csv
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

LABEL_TO_INDEX = {"NORMAL": 0, "PNEUMONIA": 1}


class ChestXrayDataset(Dataset[tuple[Tensor, int]]):
    """Load grayscale chest X-rays for one generated split."""

    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        split: str,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.dataset_root = dataset_root
        self.transform = transform
        self.samples = self._read_manifest(manifest_path, split)
        if not self.samples:
            raise ValueError(f"No samples found for split '{split}'")

    @staticmethod
    def _read_manifest(manifest_path: Path, split: str) -> list[dict[str, str]]:
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            samples = [
                row
                for row in csv.DictReader(stream)
                if row["split"] == split and row["label"] in LABEL_TO_INDEX
            ]
        unique: dict[str, dict[str, str]] = {}
        for sample in samples:
            unique.setdefault(sample["sha256"], sample)
        return sorted(unique.values(), key=lambda row: row["relative_path"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(self.dataset_root / sample["relative_path"]) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, LABEL_TO_INDEX[sample["label"]]

    def label_counts(self) -> dict[int, int]:
        """Return class counts after exact deduplication."""
        counts = {0: 0, 1: 0}
        for sample in self.samples:
            counts[LABEL_TO_INDEX[sample["label"]]] += 1
        return counts
