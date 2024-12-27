# pylint: disable=missing-function-docstring
""" PyTorch Lightning module for the Two-Tower training loop."""

from argparse import ArgumentParser
from typing import Dict, Optional

import torch
from torch import Tensor
from torchmetrics import MeanSquaredError, Metric

from src.lit_models.base_model import BaseLitModel

MAX_RATING = 5.0


class TwoTowerLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """PyTorch Lightning module for the Two-Tower model."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None):
        super().__init__(model, args)

        self.mse: Metric = MeanSquaredError()

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        """Add model-specific arguments to the parser."""
        return parser

    def forward(self, x: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """Forward pass of the model."""
        # TODO: Concatenate earlier so that we can just pass the input on here
        if x is None:
            raise ValueError("batch argument is required for TwoTowerLitModel.")
        return self.model(**{k: v for k, v in x.items() if k != "labels"})

    def _get_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute the loss for a batch."""
        denorm_preds = self(batch) * MAX_RATING
        return self.mse(denorm_preds, batch["labels"])

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._get_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._get_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        loss = self._get_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.forward(batch)
