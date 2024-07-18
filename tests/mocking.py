"""Module to define fixtures for mocking objects in tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.data_module import MovieLensDataModule


@pytest.fixture(name="mock_zip", autouse=True)
def fixture_mock_zip():
    """Fixture to mock zipfile.ZipFile.open with a test fixture file."""
    mock_csv_path = "tests/fixtures/ratings.csv"
    mock_zipfile = MagicMock()
    mock_zipfile.__enter__.return_value = mock_zipfile
    with patch(
        "src.data.data_module.zipfile.ZipFile", MagicMock(return_value=mock_zipfile)
    ):
        with open(mock_csv_path, "rb") as file:
            mock_zipfile.open = MagicMock(return_value=file)
            yield mock_zipfile


@pytest.fixture(name="data_module", autouse=True)
def fixture_data_module():
    """Create a MovieLensDataModule instance with a temporary data directory."""
    with patch(
        "src.data.data_module.MovieLensDataModule.data_dirname",
        MagicMock(return_value=Path(tempfile.mkdtemp())),
    ):
        yield MovieLensDataModule()
