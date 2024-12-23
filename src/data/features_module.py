""" PyTorch Lightning data module for the MovieLens ratings data. """

import os
import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.base_module import BaseDataModule
from src.data.features_dataset import FeaturesDataset
from src.prepare_data.download_dataset import download_and_extract_data
from src.prepare_data.features import calculate_features
from src.utils.log import logger

warnings.filterwarnings("ignore", category=FutureWarning)

COL_RENAME = {"movieId": "movie_id", "userId": "user_id"}


class FeaturesDataModule(BaseDataModule):
    """Lightning data module for the MovieLens ratings data."""

    def __init__(self, args: Optional[Dict] = None):
        super().__init__(args)

        self.movie_features_path: Path
        self.user_features_path: Path

        self.train_dataset: FeaturesDataset
        self.val_dataset: FeaturesDataset
        self.test_dataset: FeaturesDataset

        # multilingual bert tokenizer
        # self.tokenizer = transformers.BertTokenizer.from_pretrained(
        #     "bert-base-multilingual-cased"
        # )

    @property
    def movie_features_path(self) -> Path:
        """Return the path to the ratings data."""
        return self.data_dir() / "featurized" / "movie_features.parquet"

    @property
    def user_features_path(self) -> Path:
        """Return the path to the ratings data."""
        return self.data_dir() / "featurized" / "user_features.parquet"

    def prepare_data(self) -> None:
        """Download data and other preparation steps to be done only once."""
        output_dir = self.data_dir() / "featurized"
        os.makedirs(output_dir, exist_ok=True)

        if self.movie_features_path.exists() and self.user_features_path.exists():
            logger.info("Features data already exists. Skipping preparation.")
            return
        if self.rating_data_path.exists() and self.movie_data_path.exists():
            logger.info("Ratings and movie data data already exists.")
        else:
            download_and_extract_data()

        # Load data
        movies = pd.read_csv(self.movie_data_path).rename(columns=COL_RENAME)
        ratings = pd.read_csv(self.rating_data_path).rename(columns=COL_RENAME)
        ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

        # Only keep movies that have been rated
        movies = movies[movies["movie_id"].isin(ratings["movie_id"])].copy()

        logger.info("Calculating features ...")
        movie_ft, user_ft = calculate_features(ratings, movies)
        del movies, ratings

        logger.info("Saving features data to %s ...", output_dir)
        movie_ft.to_parquet(self.movie_features_path, index=False)
        user_ft.to_parquet(self.user_features_path, index=False)

    def setup(self, stage: Optional[str] = None):
        """Split the data into train, val, and test sets based on timestamps."""
        # Load data
        user_features = pd.read_parquet(self.user_features_path)
        movie_features = pd.read_parquet(self.movie_features_path)
        ratings = pd.read_csv(self.rating_data_path).rename(columns=COL_RENAME)
        ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

        # Merge ratings with user and movie features
        data = pd.merge_asof(
            ratings.sort_values("timestamp"),
            user_features.sort_values("timestamp"),
            by="user_id",
            on="timestamp",
            direction="backward",  # Use the latest snapshot before the interaction
        ).merge(movie_features, on="movie_id")

        val_size = int(len(data) * self.val_frac)
        test_size = int(len(data) * self.test_frac)

        # Sort data by timestamp to get time-based split
        data = data.sort_values("timestamp")

        train_data = data[: -(test_size + val_size)]
        val_data = data[-(test_size + val_size) : -test_size]
        test_data = data[-test_size:]

        logger.info("Train data shape: %s", train_data.shape)
        logger.info("Validation data shape: %s", val_data.shape)
        logger.info("Test data shape: %s", test_data.shape)

        # Define datasets
        self.train_dataset = FeaturesDataset(train_data)
        self.val_dataset = FeaturesDataset(val_data)
        self.test_dataset = FeaturesDataset(test_data)
