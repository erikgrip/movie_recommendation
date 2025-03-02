"""Module to define fixtures for mocking objects in tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from retrieval_model_training.data.ratings_module import RatingsDataModule
from retrieval_model_training.models.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
)

MOCK_ZIP_PATH = "tests/fixtures/sample_100.zip"


@pytest.fixture(name="ratings_data_module", autouse=True)
def fixture_ratings_data_module():
    """Create a RatingsDataModule instance with a temporary data directory."""
    tmpdir = Path(tempfile.mkdtemp())
    data_dir = tmpdir / "data"
    Path(data_dir / "extracted").mkdir(parents=True, exist_ok=True)

    with (
        patch(
            "retrieval_model_training.data.ratings_module.RatingsDataModule.data_dir",
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


@pytest.fixture(name="model")
def fixture_model(ratings_data_module):
    """Create an example LitNeuralCollaborativeFilteringModel model."""
    ratings_data_module.prepare_data()
    ratings_data_module.setup()

    args = {
        "embedding_size": 4,
        "hidden_dim": 4,
        "dropout_rate": 0.0,
    }
    return NeuralCollaborativeFilteringModel(
        num_users=ratings_data_module.num_user_labels(),
        num_movies=ratings_data_module.num_movie_labels(),
        args=args,
    )
