"""Tests for the LitRecommender class."""

import pytest
import torch

from src.lit_models.recommender import LitRecommender
from src.models.embedding_model import RecommendationModel
from tests.mocking import (  # pylint: disable=unused-import
    fixture_data_module, fixture_mock_zip)


@pytest.fixture(name="lit_model")
def fixture_lit_model(data_module):
    """Create an example LitRecommender model."""
    data_module.setup()

    model = RecommendationModel(
        num_users=data_module.num_user_labels(),
        num_movies=data_module.num_movie_labels(),
        embedding_size=4,
        hidden_dim=4,
        dropout_rate=0.0,
    )
    return LitRecommender(model=model)


def test_lit_recommender_forward(lit_model):
    """Test the forward method of LitRecommender."""
    users = torch.tensor([0])
    movies = torch.tensor([1])
    output = lit_model(users=users, movies=movies)
    # TODO: implement lit_model.forward() method
    assert output is None


def test_lit_recommender_training_step(lit_model):
    """Test the training_step method of LitRecommender."""
    train_batch = {
        "users": torch.tensor([0]),
        "movies": torch.tensor([1]),
        "ratings": torch.tensor([5.0]),
    }
    loss = lit_model.training_step(train_batch=train_batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_lit_recommender_configure_optimizers(lit_model):
    """Test the configure_optimizers method of LitRecommender."""
    optimizer = lit_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
