"""Tests for the LitRecommender class."""

# pylint: disable=unused-import

import pytest
import torch

from src.lit_models.recommender import LitRecommender
from src.models.embedding_model import RecommendationModel
from tests.mocking import fixture_data_module, fixture_mock_zip


@pytest.fixture(name="model")
def fixture_model(data_module):
    """Create an example LitRecommender model."""
    data_module.prepare_data()
    data_module.setup()

    args = {
        "embedding_size": 4,
        "hidden_dim": 4,
        "dropout_rate": 0.0,
    }
    return RecommendationModel(
        num_users=data_module.num_user_labels(),
        num_movies=data_module.num_movie_labels(),
        args=args,
    )


@pytest.fixture(name="lit_model")
def fixture_lit_model(model):
    """Create an example LitRecommender model."""
    return LitRecommender(model)


def test_lit_recommender_init(model):
    """Test the initialization of LitRecommender."""
    lit_model = LitRecommender(model=model)
    assert lit_model.optimizer_class == torch.optim.Adam
    assert lit_model.lr == 1e-3
    assert lit_model.one_cycle_max_lr is None
    assert lit_model.one_cycle_total_steps == 100
    assert not lit_model.training_step_losses


def test_lit_recommender_forward(lit_model):
    """Test the forward method of LitRecommender."""
    users = torch.tensor([0])
    movies = torch.tensor([1])
    output = lit_model(users=users, movies=movies)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1)


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
    """Test the configure_optimizers method of LitRecommender."""
    lit_model = LitRecommender(model=model, args=args)
    cfg = lit_model.configure_optimizers()
    optimizer = cfg["optimizer"]
    lr_scheduler = cfg.get("lr_scheduler")
    assert isinstance(optimizer, expected_optimizer)
    assert optimizer.defaults["lr"] == expected_lr
    if expected_lr_scheduler:
        assert isinstance(lr_scheduler, expected_lr_scheduler)
    else:
        assert lr_scheduler is None
