""" Movie recommendation model. """

# pylint: disable=arguments-differ,unused-argument

import pytorch_lightning as pl
import torch


class LitRecommender(pl.LightningModule):
    """PyTorch Lightning module for the movie recommendation model."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, x=None, **kwargs):
        """Forward pass of the model."""
        # in lightning, forward defines the prediction/inference actions

    def training_step(self, train_batch=None, batch_idx=None):
        """Training step."""
        output = self.model(train_batch["users"], train_batch["movies"])
        # Reshape the model output to match the target's shape
        output = output.squeeze()  # Removes the singleton dimension
        ratings = train_batch["ratings"].to(
            torch.float32
        )  # Assuming ratings is already 1D

        loss = torch.nn.MSELoss(output, ratings)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
