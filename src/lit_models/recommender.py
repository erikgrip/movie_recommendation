""" Movie recommendation model. """

# pylint: disable=arguments-differ,unused-argument

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.utils.log import logger

LR = 1e-3
OPTIMIZER = "Adam"
ONE_CYCLE_TOTAL_STEPS = 100


class LitRecommender(pl.LightningModule):
    """PyTorch Lightning module for the movie recommendation model."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None):
        super().__init__()
        self.model = model
        self.args = args or {}
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get("lr", LR)
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr")
        self.one_cycle_total_steps = self.args.get(
            "one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS
        )
        self.training_step_losses: list[torch.Tensor] = []

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
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
        self.log("train_loss", loss, prog_bar=True)
        self.training_step_losses.append(loss)
        return loss

    def test_step(
        self, val_batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None
    ):
        """Test step."""
        output = self(val_batch["users"], val_batch["movies"])
        output = output.squeeze()

    def on_train_end(self):
        all_losses = torch.stack(self.training_step_losses)
        logger.info(
            "Avg loss first 5 training steps: %s",
            round(torch.mean(all_losses[:5]).item(), 4),
        )
        logger.info(
            "Avg loss last 5 training steps: %s",
            round(torch.mean(all_losses[-5:]).item(), 4),
        )
        self.training_step_losses.clear()  # free memory

    def configure_optimizers(self):
        """Initialize optimizer and learning rate scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return {"optimizer": optimizer}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
