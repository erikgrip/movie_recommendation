import pytorch_lightning as pl
import torch
from torch import nn


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        pass

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        pass
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
