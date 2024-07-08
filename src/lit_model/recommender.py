""" Movie recommendation model. """

import pytorch_lightning as pl
import torch


class LitRecommender(pl.LightningModule):
    """PyTorch Lightning module for the movie recommendation model."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, x=None, **kwargs):
        """Forward pass of the model."""
        # in lightning, forward defines the prediction/inference actions

    def training_step(self, *args, batch=None, batch_idx=None, **kwargs):
        """Training step."""
        # training_step defines the train loop. It is independent of forward
        # return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
