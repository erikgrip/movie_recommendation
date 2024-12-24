"""Tests for the LitFactorizationModel class."""

# pylint: disable=unused-import

from argparse import ArgumentParser
from unittest.mock import patch

import pytest
import torch

from src.lit_models.lit_factorization import LitFactorizationModel
from src.models.factorization import FactorizationModel
from tests.mocking import fixture_ratings_data_module


@pytest.fixture(name="model")
def fixture_model(ratings_data_module):
    """Create an example LitFactorizationModel model."""
    ratings_data_module.prepare_data()
    ratings_data_module.setup()

    args = {
        "embedding_size": 4,
        "hidden_dim": 4,
        "dropout_rate": 0.0,
    }
    return FactorizationModel(
        num_users=ratings_data_module.num_user_labels(),
        num_movies=ratings_data_module.num_movie_labels(),
        args=args,
    )


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


def test_lit_recommender_init(model):
    """Test the initialization of LitFactorizationModel."""
    lit_model = LitFactorizationModel(model=model)
    assert lit_model.optimizer_class == torch.optim.Adam
    assert lit_model.lr == 1e-3
    assert lit_model.one_cycle_max_lr is None
    assert lit_model.one_cycle_total_steps == 100


def test_lit_recommender_forward(lit_model, batch):
    """Test the forward method of LitFactorizationModel."""
    output = lit_model(users=batch["user_label"], movies=batch["movie_label"])
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1)


def test_lit_recommender_training_step(lit_model, batch):
    """Test the training_step method of LitFactorizationModel."""
    loss = lit_model.training_step(train_batch=batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_lit_recommender_test_step(lit_model, batch):
    """Test the test_step method of LitFactorizationModel."""
    output = lit_model.test_step(test_batch=batch)
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
        output = lit_model.test_step(test_batch=batch)
    assert round(output["loss"].item(), 3) == 1.333  # 16.0 / 12
    assert round(output["rmse"].item(), 3) == 1.155
    assert round(output["precision"].item(), 3) == 0.6  # (0.6 + 0.6) / 2
    assert round(output["recall"].item(), 3) == 0.875  # (1 + 0.75) / 2


def test_lit_recommender_add_to_argparse():
    """Test the add_to_argparse method of LitFactorizationModel."""
    parser = ArgumentParser()
    parser = LitFactorizationModel.add_to_argparse(parser)
    args = parser.parse_args([])
    assert args.optimizer == "Adam"
    assert args.lr == 1e-3
    assert args.one_cycle_total_steps == 100


def test_lit_recommender_configure_optimizers(lit_model):
    """Test the configure_optimizers method of LitFactorizationModel."""
    cfg = lit_model.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.Optimizer)
    assert cfg.get("lr_scheduler") is None


@pytest.mark.parametrize(
    "args,expected_optimizer,expected_lr_scheduler,expected_lr",
    [
        ({"optimizer": "Adam"}, torch.optim.Adam, None, 1e-3),
        ({"optimizer": "SGD"}, torch.optim.SGD, None, 1e-3),
        ({"optimizer": "AdamW"}, torch.optim.AdamW, None, 1e-3),
        ({"lr": 0.1}, torch.optim.Adam, None, 0.1),
        (
            {"one_cycle_max_lr": 0.1},
            torch.optim.Adam,
            torch.optim.lr_scheduler.OneCycleLR,
            1e-3,
        ),
        ({"one_cycle_total_steps": 100}, torch.optim.Adam, None, 1e-3),
    ],
)
def test_lit_recommender_configure_optimizers_non_default(
    model, args, expected_optimizer, expected_lr_scheduler, expected_lr
):
    """Test the configure_optimizers method of LitFactorizationModel."""
    lit_model = LitFactorizationModel(model=model, args=args)
    cfg = lit_model.configure_optimizers()
    optimizer = cfg["optimizer"]
    lr_scheduler = cfg.get("lr_scheduler")
    assert isinstance(optimizer, expected_optimizer)
    assert optimizer.defaults["lr"] == expected_lr
    if expected_lr_scheduler:
        assert isinstance(lr_scheduler, expected_lr_scheduler)
    else:
        assert lr_scheduler is None
