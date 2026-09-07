"""Leakage-safe patient-grouped dataset splitting."""

import random
from collections import defaultdict
from collections.abc import Iterable

from pneumonia.data.models import ImageRecord

SPLIT_NAMES = ("train", "val", "test")


def duplicate_hashes(records: Iterable[ImageRecord]) -> dict[str, list[str]]:
    files_by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.is_valid:
            files_by_hash[record.sha256].append(record.relative_path)
    return {digest: sorted(paths) for digest, paths in files_by_hash.items() if len(paths) > 1}


def _allocate_groups(
    patient_ids: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = sorted(patient_ids)
    rng.shuffle(shuffled)
    count = len(shuffled)
    train_end = round(count * train_ratio)
    val_end = train_end + round(count * val_ratio)
    return {
        patient_id: ("train" if index < train_end else "val" if index < val_end else "test")
        for index, patient_id in enumerate(shuffled)
    }


def assign_grouped_splits(
    records: Iterable[ImageRecord],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, str]:
    """Assign patients to train/val/test while approximately stratifying by label."""
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    labels_by_patient: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.is_valid:
            labels_by_patient[record.patient_id].add(record.label)

    conflicting = {
        patient_id: labels for patient_id, labels in labels_by_patient.items() if len(labels) > 1
    }
    if conflicting:
        sample = sorted(conflicting)[0]
        raise ValueError(f"Patient {sample} has multiple labels: {sorted(conflicting[sample])}")

    patients_by_label: dict[str, list[str]] = defaultdict(list)
    for patient_id, labels in labels_by_patient.items():
        patients_by_label[next(iter(labels))].append(patient_id)

    assignments: dict[str, str] = {}
    for label, patient_ids in sorted(patients_by_label.items()):
        label_seed = seed + sum(ord(character) for character in label)
        assignments.update(
            _allocate_groups(
                patient_ids,
                seed=label_seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
        )
    return assignments


def assert_no_patient_leakage(assignments: dict[str, str]) -> None:
    """Validate the one-patient-to-one-split invariant."""
    invalid = {patient: split for patient, split in assignments.items() if split not in SPLIT_NAMES}
    if invalid:
        raise ValueError(f"Invalid split assignments: {invalid}")
