""" PyTorch Lightning data module for the MovieLens ratings data. """

import warnings
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=FutureWarning)

BATCH_SIZE = 32
NUM_WORKERS = 0
VAL_FRAC = 0.1
TEST_FRAC = 0.1


class BaseDataModule(pl.LightningDataModule):
    """Base DataModule class for MovieLens data."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, args: Optional[Dict] = None):
        super().__init__()
        args = args or {}
        self.batch_size = args.get("batch_size", BATCH_SIZE)
        self.num_workers = args.get("num_workers", NUM_WORKERS)
        self.val_frac = args.get("val_frac", VAL_FRAC)
        self.test_frac = args.get("test_frac", TEST_FRAC)
        self.on_gpu = (
            args.get("accelerator") in ["auto", "gpu", "cuda"]
            and torch.cuda.is_available()
        )

        self._validate_data_fractions()

        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset

    def _validate_data_fractions(self):
        """Ensure that the data fractions are valid."""
        if not 0 < self.val_frac < 1.0:
            raise ValueError("Validation fraction must be between 0 and 1.")
        if not 0 < self.test_frac < 1.0:
            raise ValueError("Test fraction must be between 0 and 1.")
        if not 0 < self.val_frac + self.test_frac < 1.0:
            raise ValueError("Validation and test fractions must sum to less than 1.0.")

    @property
    def rating_data_path(self) -> Path:
        """Return the path to the ratings data."""
        return self.data_dir() / "extracted" / "ratings.csv"

    @property
    def movie_data_path(self) -> Path:
        """Return the path to the movies data."""
        return self.data_dir() / "extracted" / "movies.csv"

    @classmethod
    def data_dir(cls) -> Path:
        """Return Path relative to where this script is stored."""
        path = Path(__file__).resolve().parents[2] / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        """Add data module arguments to the parser."""
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help=f"Number of examples in each batch (default: {BATCH_SIZE})",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=NUM_WORKERS,
            help=f"Number of workers to use for data loading (default: {NUM_WORKERS})",
        )
        parser.add_argument(
            "--val_frac",
            type=float,
            default=VAL_FRAC,
            help=f"Fraction of data to use for validation (default: {VAL_FRAC})",
        )
        parser.add_argument(
            "--test_frac",
            type=float,
            default=TEST_FRAC,
            help=f"Fraction of data to use for testing (default: {TEST_FRAC})",
        )
        return parser

    def prepare_data(self) -> None:
        """
        Use this method to do things that might write to disk,
        or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.

        Should assign `torch Dataset` objects to self.data_train,
        self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
