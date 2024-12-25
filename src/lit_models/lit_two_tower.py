import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim import Adam

from src.models.two_tower import TwoTower


class TwoTowerLitModel(pl.LightningModule):
    """PyTorch Lightning module for the Two-Tower model."""

    def __init__(
        self, 
        user_feature_dim: int, 
        movie_feature_dim: int, 
        embedding_dim: int, 
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves hyperparameters like embedding_dim, etc.

        # Instantiate the TwoTower model
        self.model = TwoTower(
            user_feature_dim=user_feature_dim, 
            movie_feature_dim=movie_feature_dim, 
            embedding_dim=embedding_dim
        )
        self.learning_rate = learning_rate

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

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer