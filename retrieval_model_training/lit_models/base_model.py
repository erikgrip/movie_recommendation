# pylint: disable=arguments-differ
""" Base class for Pytorch Lightning models. """

from argparse import ArgumentParser
from typing import Dict, Optional, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
)

LR = 1e-3
OPTIMIZER = "Adam"
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):
    """Base class for Pytorch Lightning models."""

    def __init__(self, model: torch.nn.Module, args: Optional[Dict] = None) -> None:
        super().__init__()
        self.save_hyperparameters()

        args = args or {}
        self.model = model

        optimizer: str = args.get("optimizer", OPTIMIZER)
        self.optimizer_class: Type[torch.optim.Optimizer] = getattr(
            torch.optim, optimizer
        )
        self.lr: float = args.get("lr", LR)
        self.one_cycle_max_lr: Optional[float] = args.get("one_cycle_max_lr")
        self.one_cycle_total_steps: int = args.get(
            "one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS
        )

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

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the model. This should be implemented by the child class."""
        raise NotImplementedError

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step. This should be implemented by the child class."""
        raise NotImplementedError

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step. This should be implemented by the child class."""
        raise NotImplementedError

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step. This should be implemented by the child class."""
        raise NotImplementedError

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> torch.Tensor:
        """Prediction step. This should be implemented by the child class."""
        raise NotImplementedError

    def configure_optimizers(
        self,
    ) -> Union[OptimizerLRSchedulerConfig, OptimizerConfig]:
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
