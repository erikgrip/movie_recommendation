"""Test training a model with the main function."""

import argparse
from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Trainer

from train import main


@pytest.fixture(name="overfit_batches_args")
def fixture_overfit_batches_args():
    """Return arguments to overfit batches."""
    return argparse.Namespace(
        data_class="MovieLensDataModule",
        model_class="RecommendationModel",
        overfit_batches=1.0,
        fast_dev_run=0,
        devices="auto",
        accelerator="auto",
        num_workers=0,
        max_epochs=-1,
        early_stopping=30,
    )


def test_main_overfit_flow(overfit_batches_args):
    """Test the main function flow with overfit_batches set to 1.0."""
    with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = overfit_batches_args
        with patch("train.Trainer") as mock_trainer:
            mock_trainer.return_value = MagicMock(spec=Trainer)
            _ = main()
            mock_trainer.assert_called_once()
            mock_trainer.return_value.fit.assert_called_once()
            mock_trainer.return_value.test.assert_not_called()


def test_main_overfit_learning(overfit_batches_args):
    """Test the main function learning with overfit_batches set to 1.0."""
    with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = overfit_batches_args
        trainer = main()
        assert trainer.logged_metrics["train_loss_epoch"] < 0.1
