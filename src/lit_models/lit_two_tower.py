""" PyTorch Lightning module for the Two-Tower training loop."""

from argparse import ArgumentParser
from typing import Dict, Optional

import torch
from torch.nn import functional as F

from src.lit_models.base_model import BaseLitModel


class TwoTowerLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """PyTorch Lightning module for the Two-Tower model."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None):
        super().__init__(model, args)
        self.save_hyperparameters()  # Automatically saves hyperparameters like embedding_dim, etc.

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        """Add model-specific arguments to the parser."""
        return parser

    def forward(self, x: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass of the model."""
        # TODO: Concatenate earlier so that we can just pass the input on here
        if x is None:
            raise ValueError("batch argument is required for TwoTowerLitModel.")

        user_features = x["user_pref"]
        title_embeddings = x["title_embedding"]  # Precomputed embeddings
        movie_features = torch.cat(
            [x["genres"], x["release_year"].unsqueeze(-1)], dim=1
        )

        return self.model.predict(user_features, title_embeddings, movie_features)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        preds = self.forward(batch)
        loss = F.mse_loss(preds, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # TODO: Move to base class if it stays the same
    # pylint: disable=duplicate-code
    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        preds = self.forward(batch)
        loss = F.mse_loss(preds, batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        preds = self.forward(batch)
        loss = F.mse_loss(preds, batch["labels"])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Prediction step."""
        return self.forward(batch)
