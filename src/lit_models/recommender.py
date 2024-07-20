""" Movie recommendation model. """

# pylint: disable=arguments-differ,unused-argument

import typing
from argparse import ArgumentParser
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics import (MeanSquaredError, Metric, RetrievalPrecision,
                          RetrievalRecall)

from src.utils.log import logger

LR = 1e-3
OPTIMIZER = "Adam"
ONE_CYCLE_TOTAL_STEPS = 100


class LitRecommender(pl.LightningModule):
    """PyTorch Lightning module for the movie recommendation model."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None):
        super().__init__()
        args = args or {}
        optimizer: str = args.get("optimizer", OPTIMIZER)

        self.model = model
        self.optimizer_class: typing.Type[torch.optim.Optimizer] = getattr(
            torch.optim, optimizer
        )
        self.lr: float = args.get("lr", LR)
        self.one_cycle_max_lr: Optional[float] = args.get("one_cycle_max_lr")
        self.one_cycle_total_steps: int = args.get(
            "one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS
        )
        self.training_step_losses: list[torch.Tensor] = []
        self.rmse: Metric = MeanSquaredError(squared=False)
        self.precision: Metric = RetrievalPrecision(top_k=5, empty_target_action="skip")
        self.recall: Metric = RetrievalRecall(top_k=5, empty_target_action="skip")

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        # pylint: disable=missing-function-docstring
        parser.add_argument(
            "--optimizer",
            type=str,
            default=OPTIMIZER,
            help=f"Optimizer class from torch.optim (default: {OPTIMIZER})",
        )
        parser.add_argument(
            "--lr", type=float, default=LR, help=f"learning rate (default: {LR})"
        )
        parser.add_argument(
            "--one_cycle_total_steps",
            type=int,
            default=ONE_CYCLE_TOTAL_STEPS,
            help="Total steps for the 1cycle policy",
        )
        return parser

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(users, movies)

    def training_step(
        self, train_batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Training step."""
        output = self(train_batch["users"], train_batch["movies"])
        output = output.squeeze()  # Removes the singleton dimension
        ratings = train_batch["ratings"].to(torch.float32)
        loss = F.mse_loss(output, ratings)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_step_losses.append(loss)
        return loss

    def test_step(
        self, test_batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None
    ) -> None:
        """Test step."""
        output = self(test_batch["users"], test_batch["movies"])
        y_pred = output.view(-1)
        y_true = test_batch["ratings"].to(torch.float32)

        # Calculate RMSE
        rmse = self.rmse(y_pred, y_true)
        self.log("test_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True)

        # NOTE: Don't calculate precision and recall per batch, but at the end of the epoch
        # when complete predictions for each user are available
        is_high_rating = y_true > 3.5  # 4s and 5s are considered positive

        # Calculate precision
        precision = self.precision(y_pred, is_high_rating, indexes=test_batch["users"])
        self.log(
            "test_precision", precision, on_step=False, on_epoch=True, prog_bar=True
        )

        # Calculate recall
        recall = self.recall(y_pred, is_high_rating, indexes=test_batch["users"])
        self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_end(self) -> None:
        all_losses = torch.stack(self.training_step_losses)
        logger.info(
            "Avg loss first 5 training steps: %s",
            round(torch.mean(all_losses[:5]).item(), 4),
        )
        logger.info(
            "Avg loss last 5 training steps: %s",
            round(torch.mean(all_losses[-5:]).item(), 4),
        )
        logger.info(
            "Avg loss all training steps: %s", round(torch.mean(all_losses).item(), 4)
        )
        self.training_step_losses.clear()  # free memory

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Initialize optimizer and learning rate scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)  # type: ignore
        if self.one_cycle_max_lr is None:
            return {"optimizer": optimizer}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
