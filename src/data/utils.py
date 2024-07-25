"""Utility functions for downloading and extracting the MovieLens dataset."""

import os
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
ZIP_SAVE_PATH = "data/raw/ml-latest.zip"
FILES_TO_EXTRACT = {
    "ml-latest/ratings.csv": "ratings.csv",
    "ml-latest/movies.csv": "movies.csv",
}


def download_zip() -> None:
    """Download a zip file from a URL and save it to a local path."""
    os.makedirs(ZIP_SAVE_PATH.rsplit("/", maxsplit=1)[0], exist_ok=True)
    urlretrieve(DOWNLOAD_URL, ZIP_SAVE_PATH)


def zip_exists() -> bool:
    """Check if the zip file exists in the local path."""
    return Path(ZIP_SAVE_PATH).exists()


def extract_files(directory: Path) -> None:
    """Extract the selected files from the zip file."""
    try:
        with ZipFile(ZIP_SAVE_PATH, "r") as zip_file:
            for zip_path, save_path in FILES_TO_EXTRACT.items():
                with zip_file.open(zip_path) as file, open(
                    directory / save_path, "wb"
                ) as save_file:
                    save_file.write(file.read())
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "The zip file is not found. Please download it first."
        ) from e


def extracted_files_exist(paths: list[str]) -> bool:
    """Check if the extracted files exist in the local path."""
    return all(Path(path).exists() for path in paths)
