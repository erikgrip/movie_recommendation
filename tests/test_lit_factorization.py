# pylint: disable=unused-import
"""
Tests for the LitNeuralCollaborativeFilteringModel class.
"""


from unittest.mock import patch

import pytest
import torch

from retrieval_model_training.lit_models.lit_neural_collaborative_filtering import (
    LitNeuralCollaborativeFilteringModel,
)
from tests.mocking import fixture_model, fixture_ratings_data_module


@pytest.fixture(name="lit_model")
def fixture_lit_model(model):
    """
    Create a LitNeuralCollaborativeFilteringModel instance for testing.
    """
    return LitNeuralCollaborativeFilteringModel(model)


@pytest.fixture(name="batch")
def fixture_batch():
    """
    Create a sample batch of recommendation data for testing.

    The batch contains a single user-movie interaction with corresponding labels and target rating.
    """
    yield {
        "user_label": torch.tensor([0]),
        "movie_label": torch.tensor([1]),
        "user_id": torch.tensor([2]),
        "movie_id": torch.tensor([4]),
        "target": torch.tensor([5.0]),
    }


def test_lit_recommender_forward(lit_model):
    """
    Test the forward pass of the recommendation model.

    Verifies that the model produces tensors of the expected shape and type
    when given user and movie inputs.
    """
    model_input = {"users": torch.tensor([0]), "movies": torch.tensor([1])}
    prediction = lit_model(model_input)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (1, 1)


def test_lit_recommender_training_step(lit_model, batch):
    """
    Test the training step of the recommendation model.

    Verifies that the training step produces a valid loss value when
    processing a batch of data.
    """
    loss = lit_model.training_step(batch=batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_lit_recommender_test_step(lit_model, batch):
    """
    Test the evaluation step of the recommendation model.

    Verifies that the test step produces the expected evaluation metrics
    (loss, RMSE, precision, recall) with valid values.
    """
    evaluation_metrics = lit_model.test_step(batch=batch, batch_idx=0)
    assert isinstance(evaluation_metrics, dict)
    assert evaluation_metrics.keys() == {"loss", "rmse", "precision", "recall"}
    assert evaluation_metrics["loss"].item() >= 0.0
    assert evaluation_metrics["rmse"].item() >= 0.0
    assert 0.0 <= evaluation_metrics["precision"].item() <= 1.0
    assert 0.0 <= evaluation_metrics["recall"].item() <= 1.0


def test_lit_recommender_test_step_metric_calculation(lit_model):
    """
    Test the detailed metric calculation in the test step.

    Uses a mock to control model predictions and verifies that metrics
    (loss, RMSE, precision, recall) are calculated correctly for a larger batch.
    """
    test_batch = {
        "user_label": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "movie_label": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        "user_id": torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
        "movie_id": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 15, 18, 20, 25]),
        "target": torch.tensor(
            [5.0, 3.0, 4.0, 2.0, 5.0, 4.0, 1.0, 5.0, 3.0, 4.0, 2.0, 5.0]
        ),
    }
    with patch(
        "retrieval_model_training.lit_models.lit_neural_collaborative_filtering."
        "LitNeuralCollaborativeFilteringModel.forward"
    ) as mock_forward:
        # 11 correct predictions and one off by 4
        mock_forward.return_value = torch.tensor(
            [5.0, 3.0, 4.0, 2.0, 5.0, 4.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0]
        )
        metrics = lit_model.test_step(batch=test_batch, batch_idx=0)
    assert round(metrics["loss"].item(), 3) == 1.333  # 16.0 / 12
    assert round(metrics["rmse"].item(), 3) == 1.155
    assert round(metrics["precision"].item(), 3) == 0.6  # (0.6 + 0.6) / 2
    assert round(metrics["recall"].item(), 3) == 0.875  # (1 + 0.75) / 2
