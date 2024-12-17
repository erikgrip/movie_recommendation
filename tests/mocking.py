"""Module to define fixtures for mocking objects in tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.ratings_module import RatingsDataModule

MOCK_ZIP_PATH = "tests/fixtures/sample_100.zip"


@pytest.fixture(name="ratings_data_module", autouse=True)
def fixture_ratings_data_module():
    """Create a RatingsDataModule instance with a temporary data directory."""
    tmpdir = Path(tempfile.mkdtemp())
    data_dir = tmpdir / "data"
    Path(data_dir / "extracted").mkdir(parents=True, exist_ok=True)

    with (
        patch(
            "src.data.ratings_module.RatingsDataModule.data_dir",
            MagicMock(return_value=data_dir),
        ),
    ):
        pd.read_csv("tests/fixtures/ratings.csv").to_csv(
            data_dir / "extracted" / "ratings.csv", index=False
        )
        pd.read_csv("tests/fixtures/movies.csv").to_csv(
            data_dir / "extracted" / "movies.csv", index=False
        )

        yield RatingsDataModule()
