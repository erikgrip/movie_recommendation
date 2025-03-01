import os
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from prepare_data.config import (
    DOWNLOAD_URL,
    EXTRACT_DIR,
    EXTRACTED_FILES,
    ZIP_SAVE_PATH,
)
from utils.log import logger


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


if __name__ == "__main__":
    if all(Path(ft_file).exists() for ft_file in EXTRACTED_FILES.values()):
        logger.info("Extracted files already exist.")
    else:
        if Path(ZIP_SAVE_PATH).exists():
            logger.info("Zip file already exists. Skipping download.")
        else:
            os.makedirs(os.path.dirname(ZIP_SAVE_PATH), exist_ok=True)
            logger.info("Downloading MovieLens data to %s ...", ZIP_SAVE_PATH)
            download_zip(DOWNLOAD_URL, str(ZIP_SAVE_PATH))
        logger.info("Extracting files ...")
        extract_files()
