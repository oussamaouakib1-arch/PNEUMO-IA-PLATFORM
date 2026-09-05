import json
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from pneumonia.data.cli import main, run_pipeline
from pneumonia.data.download import download_kaggle_dataset


def _create_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size=(64, 64), color=value).save(path)


def test_pipeline_writes_catalog_manifest_and_quality_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for index in range(4):
        _create_image(
            dataset / "train" / "NORMAL" / f"NORMAL2-IM-{index:04d}-0001.jpeg",
            value=20 + index,
        )
        _create_image(
            dataset / "train" / "PNEUMONIA" / f"person{index}_bacteria_1.jpeg",
            value=100 + index,
        )

    catalog = tmp_path / "output" / "catalog.csv"
    manifest = tmp_path / "output" / "manifest.csv"
    report_path = tmp_path / "output" / "report.json"
    duplicates = tmp_path / "output" / "duplicates.json"
    args = Namespace(
        input=dataset,
        catalog=catalog,
        manifest=manifest,
        report=report_path,
        duplicates=duplicates,
        seed=42,
    )

    report = run_pipeline(args)

    assert report["images_total"] == 8
    assert report["images_rejected"] == 0
    assert report["labels"] == {"NORMAL": 4, "PNEUMONIA": 4}
    assert catalog.exists()
    assert manifest.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["patients_total"] == 8
    # All generated images contain different grayscale values.
    assert json.loads(duplicates.read_text(encoding="utf-8")) == {}


def test_cli_fails_when_no_images_are_present(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No supported images"):
        main(["--input", str(tmp_path)])


def test_download_adapter_explains_missing_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_kagglehub(name: str, *args: object, **kwargs: object) -> object:
        if name == "kagglehub":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_kagglehub)

    with pytest.raises(RuntimeError, match="Install dataset dependencies"):
        download_kaggle_dataset(tmp_path)
