""" PyTorch Lightning data module for the MovieLens ratings data. """

import zipfile
from argparse import ArgumentParser
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import DataLoader

from src.data.dataset import MovieLensDataset
from src.utils.log import logger

BATCH_SIZE = 32
NUM_WORKERS = 0
TEST_FRAC = 0.2


class MovieLensDataModule(pl.LightningDataModule):
    """Lightning data module for the MovieLens ratings data."""

    data_file_name: str = "ratings.csv"
    train_dataset: MovieLensDataset
    test_dataset: MovieLensDataset
    user_label_encoder: LabelEncoder = LabelEncoder()
    movie_label_encoder: LabelEncoder = LabelEncoder()

    def __init__(self, args: Optional[Dict] = None):
        super().__init__()
        args = args or {}
        self.batch_size = args.get("batch_size", BATCH_SIZE)
        self.num_workers = args.get("num_workers", NUM_WORKERS)
        self.test_frac = args.get("test_frac", TEST_FRAC)
        self._validate_test_frac()

        self._zip_file: Path = self.data_dirname() / "ml-latest.zip"
        self._data_path: Path = self.data_dirname() / self.data_file_name

    def _validate_test_frac(self):
        if not 0.05 < self.test_frac < 0.95:
            raise ValueError("test_frac must be between 0.05 and 0.95")

    @classmethod
    def data_dirname(cls) -> Path:
        """Return Path relative to where this script is stored."""
        return Path(__file__).resolve().parents[2] / "data"

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
            "--test_frac",
            type=float,
            default=TEST_FRAC,
            help=f"Fraction of data to use for testing (default: {TEST_FRAC})",
        )
        return parser

    @property
    def zip_path(self) -> str:
        """Return the path to the ratings data zip file."""
        return str(self._zip_file)

    @property
    def data_path(self) -> str:
        """Return the path to the ratings data."""
        return str(self._data_path)

    def num_user_labels(self) -> int:
        """Return the number of unique users in the dataset."""
        try:
            classes = self.user_label_encoder.classes_
        except AttributeError as e:
            raise ValueError(
                "DataModule not yet setup. Please call `setup` first."
            ) from e
        return classes.shape[0]

    def num_movie_labels(self) -> int:
        """Return the number of unique movies in the dataset."""
        try:
            classes = self.movie_label_encoder.classes_
        except AttributeError as e:
            raise ValueError(
                "DataModule not yet setup. Please call `setup` first."
            ) from e
        return classes.shape[0]

    def prepare_data(self):
        """Download data and other preparation steps to be done only once."""
        if self._data_path.exists():
            logger.info("Using data already exctracted at %s", self._data_path)
            return

        with zipfile.ZipFile(self._zip_file, "r") as archive:
            with archive.open(f"ml-latest/{self.data_file_name}", "r") as file:
                logger.info("Writing data to %s...", self.data_path)
                pd.read_csv(TextIOWrapper(file, "utf-8")).to_csv(
                    self._data_path, index=False
                )

    def setup(self, stage: str = ""):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
        df = (
            pd.read_csv(self._data_path)
            .sort_values(by="timestamp", ascending=False)[dtypes.keys()]
            .astype(dtypes)
        )

        df["user_label"] = self.user_label_encoder.fit_transform(df.userId)
        df["movie_label"] = self.movie_label_encoder.fit_transform(df.movieId)
        df = df[["user_label", "movie_label", "rating"]]

        test_size = round(len(df) * self.test_frac)

        if stage == "fit":
            self.train_dataset = MovieLensDataset(
                *df.iloc[test_size:].to_dict(orient="list").values()
            )

        if stage == "test":
            self.test_dataset = MovieLensDataset(
                *df.iloc[:test_size].to_dict(orient="list").values()
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
