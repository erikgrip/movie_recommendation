""" Movie recommendation model. """

# pylint: disable=arguments-differ,unused-argument

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.utils.log import logger


class LitRecommender(pl.LightningModule):
    """PyTorch Lightning module for the movie recommendation model."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None):
        super().__init__()
        self.args = args or {}
        self.model = model
        self.training_step_losses: list[torch.Tensor] = []

    def forward(self, *args, x=None, **kwargs):
        """Forward pass of the model."""
        # in lightning, forward defines the prediction/inference actions

    def training_step(self, train_batch=None, batch_idx=None) -> torch.Tensor:
        """Training step."""
        output = self.model(train_batch["users"], train_batch["movies"])
        # Reshape the model output to match the target's shape
        output = output.squeeze()  # Removes the singleton dimension
        ratings = train_batch["ratings"].to(
            torch.float32
        )  # Assuming ratings is already 1D
        loss = F.mse_loss(output, ratings)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
