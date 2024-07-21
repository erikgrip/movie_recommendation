"""Test training a model with the main function."""

import argparse
from unittest.mock import MagicMock, patch

from pytorch_lightning import Trainer
from train import main


def test_main():
    """Test the main function."""
    args = argparse.Namespace(
        data_class="MovieLensDataModule",
        model_class="RecommendationModel",
        overfit_batches=1.0,
        max_epochs=1,
        devices=0,
        num_workers=20,
        early_stopping=10,
        accelerator="auto",
        fast_dev_run=0,
    )
    # Mock parser
    with (patch("train.argparse.ArgumentParser.parse_args")) as mock_parse:
        mock_parse.return_value = args
        # Mock Trainer
        with patch("train.Trainer") as mock_trainer:
            mock_trainer.return_value = MagicMock(spec=Trainer)
            main()
            mock_trainer.assert_called_once()
            mock_trainer.return_value.fit.assert_called_once()