""" Movie recommendation model. """

# pylint: disable=arguments-differ,unused-argument

import pytorch_lightning as pl
import torch

from src.models.embedding_model import RecommendationModel


class LitRecommender(pl.LightningModule):
    """PyTorch Lightning module for the movie recommendation model."""

    def __init__(self):
        super().__init__()
        self.model = RecommendationModel(
            num_users=1000, num_movies=1000, embedding_size=256, hidden_dim=256, dropout_rate=0.2
        )

    def forward(self, *args, x=None, **kwargs):
        """Forward pass of the model."""
        # in lightning, forward defines the prediction/inference actions

    def training_step(self, train_batch=None, batch_idx=None):
        """Training step."""
        output = recommendation_model(
            train_data["users"].to(device), train_data["movies"].to(device)
        )
        # Reshape the model output to match the target's shape
        output = output.squeeze()  # Removes the singleton dimension
        ratings = (
            train_data["ratings"].to(torch.float32).to(device)
        )  # Assuming ratings is already 1D

        loss = loss_func(output, ratings)
        inputs, target = train_batch
        output = self.model(inputs, target)
        loss = torch.nn.MSELoss(output, target.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
