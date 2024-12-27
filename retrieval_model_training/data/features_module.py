""" PyTorch Lightning data module for the MovieLens ratings data. """

import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from prepare_data.download_dataset import download_and_extract_data
from prepare_data.features import calculate_features
from retrieval_model_training.data.base_module import BaseDataModule
from retrieval_model_training.data.features_dataset import FeaturesDataset
from utils.data import COL_RENAME, time_split_data
from utils.log import logger

warnings.filterwarnings("ignore", category=FutureWarning)

PRETRAINED_EMBEDDING_DIM = 80


class FeaturesDataModule(BaseDataModule):
    """Lightning data module for the MovieLens ratings data."""

    def __init__(self, args: Optional[Dict] = None):
        super().__init__(args)

        args = args or {}

        self.train_dataset: FeaturesDataset
        self.val_dataset: FeaturesDataset
        self.test_dataset: FeaturesDataset

        self.pretrained_embedding_dim = args.get(
            "pretrained_embedding_dim", PRETRAINED_EMBEDDING_DIM
        )

    @property
    def movie_features_path(self) -> Path:
        """Return the path to the ratings data."""
        return self.data_dir() / "featurized" / "movie_features.parquet"

    @property
    def user_features_path(self) -> Path:
        """Return the path to the ratings data."""
        return self.data_dir() / "featurized" / "user_features.parquet"

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        parser = BaseDataModule.add_to_argparse(parser)
        parser.add_argument(
            "--pretrained_embedding_dim",
            type=int,
            default=PRETRAINED_EMBEDDING_DIM,
            help="Dimension to truncate the pretrained embeddings to.",
        )
        return parser

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

    def setup(self, stage: Optional[str] = None) -> None:
        """Split the data into train, val, and test sets based on timestamps."""
        user_ft = pd.read_parquet(self.user_features_path)
        movie_ft = pd.read_parquet(self.movie_features_path)
        ratings = pd.read_csv(self.rating_data_path).rename(columns=COL_RENAME)

        ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
        movie_ft["title_embedding"] = movie_ft["title_embedding"].apply(
            lambda x: x[: self.pretrained_embedding_dim]
        )

        # Merge ratings with user and movie features
        data = pd.merge_asof(
            ratings.sort_values("timestamp"),
            user_ft.sort_values("timestamp"),
            by="user_id",
            on="timestamp",
            direction="backward",  # Use the latest snapshot before the interaction
        ).merge(movie_ft, on="movie_id")

        train_data, val_data, test_data = time_split_data(
            data, test_frac=self.test_frac, val_frac=self.val_frac
        )

        user_ft_cols = data.columns[data.columns.str.startswith("avg_rating_")].tolist()
        movie_ft_cols = ["title_embedding", "year"] + data.columns[
            data.columns.str.startswith("is_")
        ].tolist()

        self.train_dataset = FeaturesDataset.from_pandas(
            user_features=train_data[user_ft_cols],
            movie_features=train_data[movie_ft_cols],
            target=train_data["target"],
        )
        self.val_dataset = FeaturesDataset.from_pandas(
            user_features=val_data[user_ft_cols],
            movie_features=val_data[movie_ft_cols],
            target=val_data["target"],
        )
        self.test_dataset = FeaturesDataset.from_pandas(
            user_features=test_data[user_ft_cols],
            movie_features=test_data[movie_ft_cols],
            target=test_data["target"],
        )
