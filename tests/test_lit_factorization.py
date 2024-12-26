"""Tests for the LitFactorizationModel class."""

# pylint: disable=unused-import

from argparse import ArgumentParser
from unittest.mock import patch

import pytest
import torch

from src.lit_models.lit_factorization import LitFactorizationModel
from src.models.factorization import FactorizationModel
from tests.mocking import fixture_model, fixture_ratings_data_module


@pytest.fixture(name="lit_model")
def fixture_lit_model(model):
    """Create an example LitFactorizationModel model."""
    return LitFactorizationModel(model)


@pytest.fixture(name="batch")
def fixture_batch():
    """Create an example batch of data."""
    yield {
        "user_label": torch.tensor([0]),
        "movie_label": torch.tensor([1]),
        "user_id": torch.tensor([2]),
        "movie_id": torch.tensor([4]),
        "rating": torch.tensor([5.0]),
    }


def test_lit_recommender_forward(lit_model):
    """Test the forward method of LitFactorizationModel."""
    input_ = {"users": torch.tensor([0]), "movies": torch.tensor([1])}
    output = lit_model(input_)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1)


def test_lit_recommender_training_step(lit_model, batch):
    """Test the training_step method of LitFactorizationModel."""
    loss = lit_model.training_step(batch=batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_lit_recommender_test_step(lit_model, batch):
    """Test the test_step method of LitFactorizationModel."""
    output = lit_model.test_step(batch=batch, batch_idx=0)
    assert isinstance(output, dict)
    assert output.keys() == {"loss", "rmse", "precision", "recall"}
    assert output["loss"].item() >= 0.0
    assert output["rmse"].item() >= 0.0
    assert 0.0 <= output["precision"].item() <= 1.0
    assert 0.0 <= output["recall"].item() <= 1.0


def test_lit_recommender_test_step_metric_calculation(lit_model):
    """Test the test_step_outputs method of LitFactorizationModel."""
    batch = {
        "user_label": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "movie_label": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        "user_id": torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
        "movie_id": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 15, 18, 20, 25]),
        "rating": torch.tensor(
            [5.0, 3.0, 4.0, 2.0, 5.0, 4.0, 1.0, 5.0, 3.0, 4.0, 2.0, 5.0]
        ),
    }
    with patch(
        "src.lit_models.lit_factorization.LitFactorizationModel.forward"
    ) as mock_forward:
        # 11 correct predictions and one off by 4
        mock_forward.return_value = torch.tensor(
            [5.0, 3.0, 4.0, 2.0, 5.0, 4.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0]
        )
        output = lit_model.test_step(batch=batch, batch_idx=0)
    assert round(output["loss"].item(), 3) == 1.333  # 16.0 / 12
    assert round(output["rmse"].item(), 3) == 1.155
    assert round(output["precision"].item(), 3) == 0.6  # (0.6 + 0.6) / 2
    assert round(output["recall"].item(), 3) == 0.875  # (1 + 0.75) / 2
