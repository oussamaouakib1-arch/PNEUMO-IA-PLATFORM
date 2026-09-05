from pathlib import Path

from PIL import Image

from pneumonia.data.catalog import (
    build_catalog,
    infer_label,
    infer_patient_id,
    infer_source_split,
)


def _create_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size=size, color=128).save(path)


def test_kaggle_metadata_inference() -> None:
    pneumonia = Path("train/PNEUMONIA/person1946_bacteria_4874.jpeg")
    normal = Path("test/NORMAL/NORMAL2-IM-1427-0001.jpeg")

    assert infer_patient_id(pneumonia) == "person1946"
    assert infer_patient_id(normal) == "normal2-im-1427"
    assert infer_label(pneumonia) == "PNEUMONIA"
    assert infer_source_split(normal) == "test"


def test_catalog_validates_images_and_rejects_small_files(tmp_path: Path) -> None:
    valid = tmp_path / "train" / "NORMAL" / "NORMAL2-IM-0001-0001.jpeg"
    small = tmp_path / "train" / "PNEUMONIA" / "person2_bacteria_1.jpeg"
    _create_image(valid)
    _create_image(small, size=(16, 16))

    records = build_catalog(tmp_path)

    assert len(records) == 2
    assert sum(record.is_valid for record in records) == 1
    assert {record.rejection_reason for record in records} == {None, "image_too_small"}
