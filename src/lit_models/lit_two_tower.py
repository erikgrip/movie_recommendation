""" PyTorch Lightning module for the Two-Tower training loop."""

from argparse import ArgumentParser
from typing import Dict, Optional

import torch
from torch.nn import functional as F

from src.lit_models.base_model import BaseLitModel


class TwoTowerLitModel(BaseLitModel):
    """PyTorch Lightning module for the Two-Tower model."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None):
        super().__init__(args)
        self.save_hyperparameters()  # Automatically saves hyperparameters like embedding_dim, etc.

        # Instantiate the TwoTower model
        self.model = model

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        """Add model-specific arguments to the parser."""
        return parser

    def forward(self, batch):
        """
        Forward pass for the model.
        Args:
            batch: A dictionary with keys ["user_pref", "title", "genres", "release_year", "labels"].
        Returns:
            The predicted similarity scores.
        """
        user_features = batch["user_pref"]
        title_embeddings = batch["title"]  # Precomputed BERT embeddings
        movie_features = torch.cat(
            [batch["genres"], batch["release_year"].unsqueeze(-1)], dim=1
        )

        return self.model.predict(user_features, title_embeddings, movie_features)

    def training_step(self, batch, batch_idx):
        """Training step."""
        preds = self.forward(batch)
        loss = F.mse_loss(preds, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        preds = self.forward(batch)
        loss = F.mse_loss(preds, batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        preds = self.forward(batch)
        loss = F.mse_loss(preds, batch["labels"])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
