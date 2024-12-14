import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from src.utils.log import logger

DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
ZIP_SAVE_DIR = Path("data/raw")
EXTRACT_DIR = Path("data/extracted")
PROCESSED_DIR = Path("data/processed")

ZIP_SAVE_PATH = ZIP_SAVE_DIR / "ml-latest.zip"
# path in zip: path to save
EXTRACTED_FILES = {
    "ml-latest/ratings.csv": EXTRACT_DIR / "ratings.csv",
    "ml-latest/movies.csv": EXTRACT_DIR / "movies.csv",
}
FEATURE_FILES = [PROCESSED_DIR / "movies.csv", PROCESSED_DIR / "ratings.csv"]


def download_zip(url: str, save_path: str) -> None:
    """Download a zip file from a URL and save it to a local path."""
    os.makedirs(save_path.rsplit("/", maxsplit=1)[0], exist_ok=True)
    urlretrieve(url, save_path)


def extract_files() -> None:
    """Extract the selected files from the zip file."""
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with ZipFile(ZIP_SAVE_PATH, "r") as zip_file:
        for zip_path, save_path in EXTRACTED_FILES.items():
            with zip_file.open(zip_path) as file, open(save_path, "wb") as save_file:
                save_file.write(file.read())


if all(Path(ft_file).exists() for ft_file in FEATURE_FILES):
    logger.info("Features data already exists.")
    sys.exit(0)
elif all(Path(ft_file).exists() for ft_file in EXTRACTED_FILES.values()):
    logger.info("Extracted files already exist.")
    # TODO: Extract features
else:
    if not Path(ZIP_SAVE_PATH).exists():
        os.makedirs(ZIP_SAVE_DIR, exist_ok=True)
        logger.info("Downloading MovieLens data to %s ...", ZIP_SAVE_PATH)
        download_zip(DOWNLOAD_URL, ZIP_SAVE_PATH)
    logger.info("Extracting files ...")
    extract_files()
    # TODO: Extract features

logger.info("Data extraction and preparation complete.")
logger.info("Features data saved to %s.", FEATURE_FILES)
