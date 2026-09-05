"""Shared data-pipeline models."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageRecord:
    """Metadata and validation result for one chest radiograph."""

    path: str
    relative_path: str
    source_split: str
    label: str
    patient_id: str
    extension: str
    size_bytes: int
    sha256: str
    width: int | None
    height: int | None
    mode: str | None
    is_valid: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record for tabular output."""
        return asdict(self)


SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png"}
VALID_LABELS = {"NORMAL", "PNEUMONIA"}
VALID_SOURCE_SPLITS = {"train", "val", "test", "unknown"}


def normalized_relative_path(path: Path, root: Path) -> str:
    """Return a platform-independent relative path."""
    return path.relative_to(root).as_posix()
