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

    def prepare_data(self) -> None:
        """Download data and other preparation steps to be done only once."""
        output_dir = self.data_dir() / "featurized"
        movie_features_path = output_dir / "movie_features.parquet"
        user_features_path = output_dir / "user_features.parquet"
        os.makedirs(output_dir, exist_ok=True)

        if movie_features_path.exists() and user_features_path.exists():
            logger.info("Features data already exists. Skipping preparation.")
            return
        if self.rating_data_path.exists() and self.movie_data_path.exists():
            logger.info("Ratings and movie data data already exists.")
        else:
            download_and_extract_data()

        # Load data
        col_rename = {"movieId": "movie_id", "userId": "user_id"}
        movies = pd.read_csv(self.movie_data_path).rename(columns=col_rename)
        ratings = pd.read_csv(self.rating_data_path).rename(columns=col_rename)

        # Only keep movies that have been rated
        movies = movies[movies["movie_id"].isin(ratings["movie_id"])].copy()

        # ---------------------
        # TODO: Drop sampling down
        logger.info("Sampling data ...")
        sample_users = ratings["user_id"].unique()[:1_000]
        ratings = ratings[ratings["user_id"].isin(sample_users)]
        movies = movies[movies["movie_id"].isin(ratings["movie_id"])]
        # ---------------------

        movie_ft, user_ft = calculate_features(ratings, movies)
        del movies, ratings

        logger.info("Saving features data ...")

        self.movie_features_path = output_dir / "movie_features.parquet"
        self.user_features_path = output_dir / "user_features.parquet"
        movie_ft.to_parquet(self.movie_features_path, index=False)
        user_ft.to_parquet(self.user_features_path, index=False)

    def setup(self, stage: Optional[str] = None):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        # TODO: Implement
        raise NotImplementedError("Setup is not implemented for FeaturesDataModule.")
