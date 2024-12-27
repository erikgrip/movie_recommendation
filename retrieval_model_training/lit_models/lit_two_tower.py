# pylint: disable=missing-function-docstring
""" PyTorch Lightning module for the Two-Tower training loop."""

from argparse import ArgumentParser
from typing import Dict, Optional

import torch
from torch import Tensor
from torchmetrics import MeanSquaredError, Metric

from retrieval_model_training.lit_models.base_model import BaseLitModel

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

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        """Forward pass of the model."""
        return self.model(**{k: v for k, v in x.items() if k != "target"})

    def _batch_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute the loss for a batch."""
        denorm_preds = self(batch) * MAX_RATING
        return self.mse(denorm_preds, batch["target"])

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._batch_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._batch_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        loss = self._batch_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.forward(batch)
