"""Write deterministic phase 3 artifacts."""

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pneumonia.data.models import ImageRecord


def write_catalog(records: Sequence[ImageRecord], output_path: Path) -> None:
    """Write the image catalog as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    if not rows:
        raise ValueError("Cannot write an empty catalog.")
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_split_manifest(
    records: Sequence[ImageRecord],
    assignments: Mapping[str, str],
    output_path: Path,
) -> None:
    """Write valid images and their generated split."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["relative_path", "label", "patient_id", "split", "sha256"]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if record.is_valid:
                writer.writerow(
                    {
                        "relative_path": record.relative_path,
                        "label": record.label,
                        "patient_id": record.patient_id,
                        "split": assignments[record.patient_id],
                        "sha256": record.sha256,
                    }
                )


def write_json(payload: Any, output_path: Path) -> None:
    """Write a human-readable deterministic JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
