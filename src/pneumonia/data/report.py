"""Phase 3 report generation."""

from collections import Counter
from collections.abc import Iterable
from typing import Any

from pneumonia.data.models import ImageRecord
from pneumonia.data.split import duplicate_hashes


def build_report(
    records: Iterable[ImageRecord],
    assignments: dict[str, str],
) -> dict[str, Any]:
    """Summarize quality, class balance, duplicates, and generated splits."""
    catalog = list(records)
    valid = [record for record in catalog if record.is_valid]
    invalid = [record for record in catalog if not record.is_valid]
    duplicate_groups = duplicate_hashes(valid)

    split_images = Counter(assignments[record.patient_id] for record in valid)
    split_patients = Counter(assignments.values())

    return {
        "images_total": len(catalog),
        "images_valid": len(valid),
        "images_rejected": len(invalid),
        "patients_total": len(assignments),
        "labels": dict(sorted(Counter(record.label for record in valid).items())),
        "source_splits": dict(sorted(Counter(record.source_split for record in valid).items())),
        "generated_split_images": dict(sorted(split_images.items())),
        "generated_split_patients": dict(sorted(split_patients.items())),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": sum(len(paths) - 1 for paths in duplicate_groups.values()),
        "rejection_reasons": dict(
            sorted(Counter(record.rejection_reason for record in invalid).items())
        ),
    }
