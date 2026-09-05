"""Download the public Kaggle chest X-ray dataset into the project."""

from pathlib import Path

from pneumonia.data.download import download_kaggle_dataset

if __name__ == "__main__":
    location = download_kaggle_dataset(Path("data/raw/kaggle"))
    print(location)
