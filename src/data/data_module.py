""" PyTorch Lightning data module for the MovieLens ratings data. """

import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import DataLoader

from src.data.dataset import MovieLensDataset


class MovieLensDataModule(pl.LightningDataModule):
    """Lightning data module for the MovieLens ratings data."""

    def __init__(self, test_frac: float = 0.1):
        super().__init__()
        self.test_frac = test_frac
        self.train_dataset: Union[MovieLensDataset, None] = None
        self.test_dataset: Union[MovieLensDataset, None] = None
        self._validate_test_frac()
        self._zip_path = self.data_dirname() / "ml-latest.zip"
        self._data_path = self.data_dirname() / "ratings.csv"
        self._user_labels_path = self.data_dirname() / "user_labels.npy"
        self._movie_labels_path = self.data_dirname() / "movie_labels.npy"

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
        return str(self._zip_path)

    @property
    def data_path(self) -> str:
        """Return the path to the ratings data."""
        return str(self._data_path)

    @property
    def user_labels_path(self) -> str:
        """Return the path to the user labels."""
        return str(self._user_labels_path)

    @property
    def movie_labels_path(self) -> str:
        """Return the path to the movie labels."""
        return str(self._movie_labels_path)

    def prepare_data(self):
        """Download data and other preparation steps to be done only once."""
        lbl_user = LabelEncoder()
        lbl_movie = LabelEncoder()

        with zipfile.ZipFile(self._data_path) as archive:
            print(archive.namelist())
            with archive.open(self.data_path) as file:
                print(f"Writing data to {self.data_path}...")
                df = pd.read_csv(TextIOWrapper(file, "utf-8"))
                df["user_label"] = lbl_user.fit_transform(df.userId)
                df["movie_label"] = lbl_movie.fit_transform(df.movieId)
                df.to_csv(self._data_path, index=False)

        # Save the label encoders
        print(f"Saving user label encoder to {self.user_labels_path}...")
        np.save(self._user_labels_path, lbl_user.classes_)
        print(f"Saving movie label encoder to {self.movie_labels_path}...")
        np.save(self._movie_labels_path, lbl_movie.classes_)

    def setup(self, stage: str):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        dtypes = {"user_label": "int32", "movie_label": "int32", "rating": "float32"}
        df = (
            pd.read_csv(self._data_path)
            .sort_values(by="timestamp", ascending=False)[dtypes.keys()]
            .astype(dtypes)
        )
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
