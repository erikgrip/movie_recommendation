""" PyTorch Lightning data module for the MovieLens ratings data. """

import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd  # type: ignore
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import DataLoader

from src.data.dataset import MovieLensDataset
from src.utils.log import logger


class MovieLensDataModule(pl.LightningDataModule):
    """Lightning data module for the MovieLens ratings data."""

    data_file_name = "ratings.csv"

    def __init__(self, test_frac: float = 0.1, args: Optional[Dict] = None):
        super().__init__()
        self.test_frac = test_frac
        self._validate_test_frac()
        self.train_dataset: Union[MovieLensDataset, None] = None
        self.test_dataset: Union[MovieLensDataset, None] = None
        self._zip_file: Path = self.data_dirname() / "ml-latest.zip"
        self._data_path: Path = self.data_dirname() / self.data_file_name
        self.user_label_encoder: LabelEncoder = LabelEncoder()
        self.movie_label_encoder: LabelEncoder = LabelEncoder()

    def _validate_test_frac(self):
        if not 0.05 < self.test_frac < 0.95:
            raise ValueError("test_frac must be between 0.05 and 0.95")

    @classmethod
    def data_dirname(cls) -> Path:
        """Return Path relative to where this script is stored."""
        return Path(__file__).resolve().parents[2] / "data"

    @property
    def zip_path(self) -> str:
        """Return the path to the ratings data zip file."""
        return str(self._zip_file)

    @property
    def data_path(self) -> str:
        """Return the path to the ratings data."""
        return str(self._data_path)

    def num_user_labels(self):
        """Return the number of unique users in the dataset."""
        try:
            classes = self.user_label_encoder.classes_
        except AttributeError as e:
            raise ValueError(
                "DataModule not yet setup. Please call `setup` first."
            ) from e
        return classes.shape[0]

    def num_movie_labels(self):
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

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, shuffle=False)
