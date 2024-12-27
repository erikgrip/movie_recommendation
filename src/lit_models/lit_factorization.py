# pylint: disable=unused-argument
""" Movie recommendation model. """

from typing import Dict, Optional

import torch
from torchmetrics import MeanSquaredError, Metric
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall

from src.lit_models.base_model import BaseLitModel


class LitFactorizationModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """PyTorch Lightning module for the movie recommendation model."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None):
        super().__init__(model, args)
        args = args or {}

        self.mse: Metric = MeanSquaredError()
        self.rmse: Metric = MeanSquaredError(squared=False)
        self.precision: Metric = RetrievalPrecision(
            top_k=5, empty_target_action="skip", adaptive_k=True
        )
        self.recall: Metric = RetrievalRecall(top_k=5, empty_target_action="skip")

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(**x)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        y_pred = self(
            {"users": batch["user_label"], "movies": batch["movie_label"]}
        ).view(-1)
        y_true = batch["label"]

        loss = self.mse(y_pred, y_true)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        y_pred = self(
            {"users": batch["user_label"], "movies": batch["movie_label"]}
        ).view(-1)
        y_true = batch["label"]

        # Calculate loss
        loss = self.mse(y_pred, y_true)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        y_pred = self(
            {"users": batch["user_label"], "movies": batch["movie_label"]}
        ).view(-1)
        y_true = batch["label"]

        # Calculate loss
        loss = self.mse(y_pred, y_true)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Calculate RMSE
        rmse = self.rmse(y_pred, y_true)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)

        # NOTE: Don't calculate precision and recall per batch, but at the end of the epoch
        # when complete predictions for each user are available
        is_high_rating = y_true > 3.5  # 4s and 5s are considered positive

        # Calculate precision
        precision = self.precision(y_pred, is_high_rating, indexes=batch["user_label"])
        self.log(
            "test_precision", precision, on_step=False, on_epoch=True, prog_bar=True
        )

        # Calculate recall
        recall = self.recall(y_pred, is_high_rating, indexes=batch["user_label"])
        self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "rmse": rmse, "precision": precision, "recall": recall}

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Prediction step."""
        return self(batch["user_label"], batch["movie_label"]).view(-1)
