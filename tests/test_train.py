"""Test training a model with the train.py main function."""

# pylint: disable=unused-import

import argparse
from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Trainer

from tests.mocking import fixture_ratings_data_module
from train import main


@pytest.fixture(name="args")
def fixture_args():
    """Return arguments to overfit batches."""
    return argparse.Namespace(
        data_class="RatingsDataModule",
        model_class="RecommendationModel",
        overfit_batches=0.0,
        fast_dev_run=0,
        devices="auto",
        accelerator="auto",
        num_workers=0,
        max_epochs=-1,
        early_stopping=100,
    )


def test_trainer_calls(args):
    """Test the main function flow."""
    with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = args
        with patch("train.Trainer") as mock_trainer:
            mock_trainer.return_value = MagicMock(spec=Trainer)
            _ = main()
            mock_trainer.assert_called_once()
            mock_trainer.return_value.fit.assert_called_once()
            mock_trainer.return_value.test.assert_called_once()


def test_overfit_trainer_calls(args):
    """Test the main function flow with overfit_batches set to 1.0."""
    args.overfit_batches = 1.0
    with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = args
        with patch("train.Trainer") as mock_trainer:
            mock_trainer.return_value = MagicMock(spec=Trainer)
            _ = main()
            mock_trainer.assert_called_once()
            mock_trainer.return_value.fit.assert_called_once()
            mock_trainer.return_value.test.assert_not_called()


def test_overfit_loss_decreases(args):
    """Test that the loss decreases when overfitting batches."""
    args.overfit_batches = 1.0
    with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = args
        trainer = main()
        assert trainer.logged_metrics["train_loss_epoch"] < 0.1
