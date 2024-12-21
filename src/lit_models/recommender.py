""" Movie recommendation model. """

# pylint: disable=arguments-differ,unused-argument

import typing
from argparse import ArgumentParser
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics import MeanSquaredError, Metric
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall

from src.utils.log import logger

LR = 1e-3
OPTIMIZER = "Adam"
ONE_CYCLE_TOTAL_STEPS = 100


class LitRecommender(
    pl.LightningModule
):  # pylint: disable=too-many-instance-attributes
    """PyTorch Lightning module for the movie recommendation model."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None):
        super().__init__()
        args = args or {}
        self.model = model

        optimizer: str = args.get("optimizer", OPTIMIZER)
        self.optimizer_class: typing.Type[torch.optim.Optimizer] = getattr(
            torch.optim, optimizer
        )
        self.lr: float = args.get("lr", LR)
        self.one_cycle_max_lr: Optional[float] = args.get("one_cycle_max_lr")
        self.one_cycle_total_steps: int = args.get(
            "one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS
        )

        self.mse: Metric = MeanSquaredError()
        self.rmse: Metric = MeanSquaredError(squared=False)
        self.precision: Metric = RetrievalPrecision(
            top_k=5, empty_target_action="skip", adaptive_k=True
        )
        self.recall: Metric = RetrievalRecall(top_k=5, empty_target_action="skip")
        self.predict_step_outputs: list[Dict[str, torch.Tensor]] = []

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        """Add model-specific arguments to the parser."""
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
        y_pred = self(train_batch["user_label"], train_batch["movie_label"]).view(-1)
        y_true = train_batch["rating"]

        loss = self.mse(y_pred, y_true)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, val_batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Validation step."""
        y_pred = self(val_batch["user_label"], val_batch["movie_label"]).view(-1)
        y_true = val_batch["rating"]

        # Calculate loss
        loss = self.mse(y_pred, y_true)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self, test_batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        y_pred = self(test_batch["user_label"], test_batch["movie_label"]).view(-1)
        y_true = test_batch["rating"]

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
        precision = self.precision(
            y_pred, is_high_rating, indexes=test_batch["user_label"]
        )
        self.log(
            "test_precision", precision, on_step=False, on_epoch=True, prog_bar=True
        )

        # Calculate recall
        recall = self.recall(y_pred, is_high_rating, indexes=test_batch["user_label"])
        self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "rmse": rmse, "precision": precision, "recall": recall}

    def predict_step(
        self,
        pred_batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: int = 0,
    ) -> None:
        """Prediction step."""
        self.predict_step_outputs.append(
            {
                "user_label": pred_batch["user_label"],
                "movie_label": pred_batch["movie_label"],
                "user_id": pred_batch["user_id"],
                "movie_id": pred_batch["movie_id"],
            }
        )

    def on_predict_end(self) -> None:
        """Prediction step."""

        def concat_results(key: str) -> torch.Tensor:
            return torch.cat([x[key] for x in self.predict_step_outputs], dim=0)

        # Concat results from all batches
        user_labels = concat_results("user_label")
        user_ids = concat_results("user_id")

        # We want to predict ratings for all movies for a single user
        all_movie_labels = torch.tensor(
            list(range(self.model.num_movies)),
            dtype=torch.long,
            device=user_labels.device,
        )

        # Randomly sample a user to show recommendations for
        random_user_label = np.random.choice(user_labels.cpu().numpy())
        random_user_id = user_ids[user_labels == random_user_label][0].cpu().numpy()
        user = (
            torch.tensor(random_user_label, dtype=torch.long)
            .to(user_labels.device)
            .repeat(all_movie_labels.size(0))
        )

        preds = self(user, all_movie_labels).view(-1)
        preds_df = pd.DataFrame(
            {
                "user_label": user.cpu(),
                "movie_label": all_movie_labels.cpu().numpy(),
                "movie_id": (
                    self.trainer.datamodule.movie_label_encoder.inverse_transform(  # type: ignore
                        all_movie_labels.cpu().numpy()
                    )
                ),
                "pred": preds.cpu().numpy(),
            }
        )

        # Show history and top 5 recommendations for a random user
        ratings = pd.read_csv(self.trainer.datamodule.rating_data_path)  # type: ignore
        ratings = ratings[ratings["userId"] == random_user_id]
        movie_meta = pd.read_csv(self.trainer.datamodule.movie_data_path)  # type: ignore

        user_history = ratings.merge(movie_meta, on="movieId")[
            ["title", "genres", "rating"]
        ].sort_values("rating", ascending=False)
        logger.info("User top 5 movies:\n%s", user_history.head(5))
        logger.info("User bottom 5 movies:\n%s", user_history.tail(5))
        top_5_rec = (
            movie_meta.merge(
                preds_df[
                    ~preds_df["movie_id"].isin(ratings["movieId"])
                ],  # unseen movies
                left_on="movieId",
                right_on="movie_id",
            )
            .sort_values(by="pred", ascending=False)
            .head(5)
        )
        logger.info(
            "Top 5 recommendations:\n%s", top_5_rec[["title", "genres", "pred"]]
        )

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Initialize optimizer and learning rate scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)  # type: ignore
        if self.one_cycle_max_lr is None:
            return {"optimizer": optimizer}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
