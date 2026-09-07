"""Build a validated, reproducible catalog of chest X-ray images."""

import hashlib
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from pneumonia.data.models import (
    SUPPORTED_EXTENSIONS,
    VALID_LABELS,
    ImageRecord,
    normalized_relative_path,
)

LOGGER = logging.getLogger(__name__)

_PERSON_PATTERN = re.compile(r"^(person\d+)", re.IGNORECASE)
_NORMAL_PATTERN = re.compile(r"^((?:NORMAL\d*)-IM-\d+)", re.IGNORECASE)


def discover_images(root: Path) -> list[Path]:
    """Find supported image files under a dataset root."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def infer_patient_id(path: Path) -> str:
    """Infer a stable patient group from common Kaggle filename formats."""
    stem = path.stem
    person_match = _PERSON_PATTERN.match(stem)
    if person_match:
        return person_match.group(1).lower()

    normal_match = _NORMAL_PATTERN.match(stem)
    if normal_match:
        return normal_match.group(1).lower()

    # Unknown naming schemes remain isolated by file to prevent accidental merging.
    return f"file-{hashlib.sha256(stem.encode('utf-8')).hexdigest()[:16]}"


def infer_label(path: Path) -> str:
    """Infer NORMAL or PNEUMONIA from the directory hierarchy."""
    for part in reversed(path.parts):
        label = part.upper()
        if label in VALID_LABELS:
            return label
    return "UNKNOWN"


def infer_source_split(path: Path) -> str:
    """Infer the dataset's original split, when present."""
    for part in reversed(path.parts):
        split = part.lower()
        if split in {"train", "val", "test"}:
            return split
    return "unknown"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int | None, int | None, str | None, str | None]:
    """Validate an image and return dimensions, mode, and rejection reason."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        if width < 32 or height < 32:
            return width, height, mode, "image_too_small"
        return width, height, mode, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, None, None, f"invalid_image:{type(exc).__name__}"


def build_record(path: Path, root: Path) -> ImageRecord:
    """Create one catalog record."""
    width, height, mode, rejection_reason = inspect_image(path)
    label = infer_label(path)
    if label == "UNKNOWN" and rejection_reason is None:
        rejection_reason = "unknown_label"

    return ImageRecord(
        path=str(path.resolve()),
        relative_path=normalized_relative_path(path, root),
        source_split=infer_source_split(path),
        label=label,
        patient_id=infer_patient_id(path),
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        width=width,
        height=height,
        mode=mode,
        is_valid=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


def build_catalog(root: Path, paths: Iterable[Path] | None = None) -> list[ImageRecord]:
    """Inspect every image and return a deterministic catalog."""
    image_paths = list(paths) if paths is not None else discover_images(root)
    LOGGER.info("Cataloging %d image(s) from %s", len(image_paths), root)
    return [build_record(path, root) for path in image_paths]
