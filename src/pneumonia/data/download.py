"""Dataset download adapter."""

from pathlib import Path


def download_kaggle_dataset(
    target_dir: Path,
    handle: str = "paultimothymooney/chest-xray-pneumonia",
) -> Path:
    """Download the public Kaggle dataset and return its cached location."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            'Install dataset dependencies first: python -m pip install -e ".[data]"'
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(kagglehub.dataset_download(handle, output_dir=str(target_dir.resolve())))
    return downloaded
