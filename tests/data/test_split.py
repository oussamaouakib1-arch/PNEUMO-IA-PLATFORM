import pytest

from pneumonia.data.models import ImageRecord
from pneumonia.data.split import (
    assert_no_patient_leakage,
    assign_grouped_splits,
    duplicate_hashes,
)


def _record(patient_id: str, label: str, digest: str, name: str) -> ImageRecord:
    return ImageRecord(
        path=name,
        relative_path=name,
        source_split="train",
        label=label,
        patient_id=patient_id,
        extension=".jpeg",
        size_bytes=100,
        sha256=digest,
        width=64,
        height=64,
        mode="L",
        is_valid=True,
        rejection_reason=None,
    )


def test_patient_images_always_share_one_generated_split() -> None:
    records = [
        _record("normal-1", "NORMAL", "a", "normal-1-a.jpeg"),
        _record("normal-1", "NORMAL", "b", "normal-1-b.jpeg"),
        _record("normal-2", "NORMAL", "c", "normal-2.jpeg"),
        _record("pneumonia-1", "PNEUMONIA", "d", "pneumonia-1.jpeg"),
        _record("pneumonia-2", "PNEUMONIA", "e", "pneumonia-2.jpeg"),
    ]

    first = assign_grouped_splits(records, seed=42)
    second = assign_grouped_splits(records, seed=42)

    assert first == second
    assert set(first) == {"normal-1", "normal-2", "pneumonia-1", "pneumonia-2"}


def test_exact_duplicates_are_reported() -> None:
    records = [
        _record("patient-1", "NORMAL", "same", "one.jpeg"),
        _record("patient-2", "NORMAL", "same", "two.jpeg"),
        _record("patient-3", "NORMAL", "unique", "three.jpeg"),
    ]

    assert duplicate_hashes(records) == {"same": ["one.jpeg", "two.jpeg"]}


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValueError, match="less than 1"):
        assign_grouped_splits([], train_ratio=0.9, val_ratio=0.2)


def test_conflicting_patient_labels_are_rejected() -> None:
    records = [
        _record("same-patient", "NORMAL", "one", "one.jpeg"),
        _record("same-patient", "PNEUMONIA", "two", "two.jpeg"),
    ]

    with pytest.raises(ValueError, match="multiple labels"):
        assign_grouped_splits(records)


def test_invalid_split_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid split"):
        assert_no_patient_leakage({"patient": "validation"})
