# pylint: disable=missing-function-docstring,unused-import
""" Tests for the BaseLitModel class. """

from argparse import ArgumentParser

import pytest
import torch

from retrieval_model_training.lit_models.base_model import BaseLitModel
from tests.mocking import fixture_model, fixture_ratings_data_module


def test_lit_recommender_init(model):
    """Test the initialization of LitNeuralCollaborativeFilteringModel."""
    lit_model = BaseLitModel(model)
    assert lit_model.optimizer_class == torch.optim.Adam
    assert lit_model.lr == 1e-3
    assert lit_model.one_cycle_max_lr is None
    assert lit_model.one_cycle_total_steps == 100


def test_lit_recommender_add_to_argparse():
    """Test the add_to_argparse method of BaseLitModel."""
    parser = ArgumentParser()
    parser = BaseLitModel.add_to_argparse(parser)
    args = parser.parse_args([])
    assert args.optimizer == "Adam"
    assert args.lr == 1e-3
    assert args.one_cycle_total_steps == 100


def test_lit_recommender_configure_optimizers(model):
    """Test the configure_optimizers method of BaseLitModel."""
    lit_model = BaseLitModel(model)
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
    """Test the configure_optimizers method of BaseLitModel."""
    lit_model = BaseLitModel(model, args)

    cfg = lit_model.configure_optimizers()
    optimizer = cfg["optimizer"]
    lr_scheduler = cfg.get("lr_scheduler")
    assert isinstance(optimizer, expected_optimizer)
    assert optimizer.defaults["lr"] == expected_lr
    if expected_lr_scheduler:
        assert isinstance(lr_scheduler, expected_lr_scheduler)
    else:
        assert lr_scheduler is None
