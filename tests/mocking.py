"""Module to define fixtures for mocking objects in tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.data_module import MovieLensDataModule

MOCK_ZIP_PATH = "tests/fixtures/sample_100.zip"


@pytest.fixture(name="data_module", autouse=True)
def fixture_data_module():
    """Create a MovieLensDataModule instance with a temporary data directory."""
    tmpdir = Path(tempfile.mkdtemp())
    with (
        patch(
            "src.data.data_module.MovieLensDataModule.data_dirname",
            MagicMock(return_value=Path(tmpdir) / "data"),
        ),
        patch("src.data.utils.ZIP_SAVE_PATH", MOCK_ZIP_PATH),
    ):
        os.mkdir(tmpdir / "data")
        yield MovieLensDataModule()
