"""Module to define fixtures for mocking objects in tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.ratings_module import RatingsDataModule

MOCK_ZIP_PATH = "tests/fixtures/sample_100.zip"


@pytest.fixture(name="ratings_data_module", autouse=True)
def fixture_ratings_data_module():
    """Create a RatingsDataModule instance with a temporary data directory."""
    tmpdir = Path(tempfile.mkdtemp())
    with (
        patch(
            "src.data.ratings_module.RatingsDataModule.data_dirname",
            MagicMock(return_value=Path(tmpdir) / "data"),
        ),
    ):
        os.mkdir(tmpdir / "data")
        yield RatingsDataModule()
