""" PyTorch Lightning data module for the MovieLens ratings data. """

import zipfile
from io import TextIOWrapper
from typing import Union

import pandas as pd  # type: ignore
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import DataLoader

from src.data.dataset import MovieLensDataset


class MovieLensDataModule(pl.LightningDataModule):
    """Lightning data module for the MovieLens ratings data."""

    data_path = "data/ml-latest.zip"
    output_path = "data/ratings.csv"

    def __init__(self, test_frac: float = 0.1):
        super().__init__()
        self.test_frac = test_frac
        self.train_dataset: Union[MovieLensDataset, None] = None
        self.test_dataset: Union[MovieLensDataset, None] = None
        self._validate_test_frac()

    def _validate_test_frac(self):
        if not 0.05 < self.test_frac < 0.95:
            raise ValueError("test_frac must be between 0.05 and 0.95")

    def prepare_data(self):
        """Download data and other preparation steps to be done only once."""
        with zipfile.ZipFile(self.data_path) as archive:
            print(archive.namelist())
            with archive.open(self.data_path) as file:
                print(f"Writing data to {self.output_path}...")
                pd.read_csv(TextIOWrapper(file, "utf-8")).to_csv(
                    self.output_path, index=False
                )

    def setup(self, stage: str):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
        df = (
            pd.read_csv(self.output_path)
            .sort_values(by="timestamp", ascending=False)[dtypes.keys()]
            .astype(dtypes)
        )
        # Encode user and movie IDs
        df["user_label"] = LabelEncoder().fit_transform(df.userId.values)
        df["movie_label"] = LabelEncoder().fit_transform(df.movieId.values)
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
