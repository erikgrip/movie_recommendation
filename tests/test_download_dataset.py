"""Tests for the data.utils module."""

import os
import zipfile
from unittest.mock import patch

import pytest

from prepare_data.download import EXTRACTED_FILES, download_zip, extract_files


@pytest.fixture(scope="function", name="zip_save_path")
def mock_zip_save_path(tmp_path_factory):
    """Fixture to create a mock zip file and return its path."""
    zip_file_path = tmp_path_factory.mktemp("data").joinpath("ml-latest.zip")
    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
        for zip_path in EXTRACTED_FILES:
            zip_file.writestr(zip_path, f"Mock zip file: {zip_path}")

    return zip_file_path


@pytest.fixture(name="mock_extracted_files")
def fixture_mock_extracted_files(tmp_path_factory):
    """Fixture to create mock extracted files and return their paths."""
    extracted_files = {
        k: tmp_path_factory.mktemp("data").joinpath(v)
        for k, v in EXTRACTED_FILES.items()
    }

    for save_path in extracted_files.values():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("Mock extracted file")

    return extracted_files


def test_download_zip(tmp_path, monkeypatch):
    """Test the download_zip function."""
    tmp_save_path = tmp_path.joinpath("ml-latest.zip")

    def mock_urlretrieve(url, save_path):  # pylint: disable=unused-argument
        with open(tmp_save_path, "w", encoding="utf-8") as file:
            file.write("Mock zip file")

    monkeypatch.setattr("prepare_data.download.urlretrieve", mock_urlretrieve)
    download_zip("some_url", str(tmp_save_path))

    assert tmp_save_path.exists()
    assert tmp_save_path.read_text() == "Mock zip file"


def test_extract_files(tmp_path, zip_save_path):
    """Test the extract_files function."""
    os.mkdir(tmp_path.joinpath("data"))

    with (
        patch("prepare_data.download.ZIP_SAVE_PATH", zip_save_path),
        patch(
            "prepare_data.download.EXTRACTED_FILES",
            {
                "ml-latest/ratings.csv": tmp_path / "ratings.csv",
                "ml-latest/movies.csv": tmp_path / "movies.csv",
            },
        ) as mock_extracted_files,
    ):
        extract_files()

        for save_path in mock_extracted_files.values():
            assert save_path.exists()


def test_extract_files_when_zip_file_not_found(tmp_path):
    """Test the extract_files function when the zip file is not found."""
    with patch(
        "prepare_data.download.ZIP_SAVE_PATH", tmp_path.joinpath("nonexistent.zip")
    ):
        with pytest.raises(FileNotFoundError):
            extract_files()
