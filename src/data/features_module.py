""" PyTorch Lightning data module for the MovieLens ratings data. """

import os
import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.base_module import BaseDataModule
from src.data.features_dataset import FeaturesDataset
from src.utils import features
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

    def _movie_features(
        self, movies: pd.DataFrame, ratings: pd.DataFrame
    ) -> pd.DataFrame:
        """Create movie features."""
        logger.info("Creating movie features ...")
        movies["genres"] = features.movie_genres_to_list(movies["genres"])
        movies["year"] = features.extract_movie_release_year(movies["title"])
        movies = features.impute_missing_year(movies, ratings)
        movies["title"] = features.clean_movie_titles(movies["title"])
        return movies

    def _user_features(
        self, movies: pd.DataFrame, ratings: pd.DataFrame
    ) -> pd.DataFrame:
        """Create user features."""
        logger.info("Creating user features ...")
        genre_dummies = features.genre_dummies(movies)
        return features.user_genre_avg_ratings(ratings, genre_dummies)

    def prepare_data(self) -> None:
        """Download data and other preparation steps to be done only once."""
        features_dir = self.data_dir() / "features_data_module"
        os.makedirs(features_dir, exist_ok=True)

        if (features_dir / "movie_features.parquet").exists() and (
            features_dir / "user_features.parquet"
        ).exists():
            logger.info("Features data already exists. Skipping preparation.")
            return

        # Load data
        col_rename = {"movieId": "movie_id", "userId": "user_id"}
        movies = pd.read_csv(self.data_dir() / "extracted/movies.csv").rename(
            columns=col_rename
        )
        ratings = pd.read_csv(self.data_dir() / "extracted/ratings.csv").rename(
            columns=col_rename
        )

        # ---------------------
        # TODO: Drop sampling down
        logger.info("Sampling data ...")
        sample_users = ratings["user_id"].unique()[:1_000]
        ratings = ratings[ratings["user_id"].isin(sample_users)]
        movies = movies[movies["movie_id"].isin(ratings["movie_id"])]
        # ---------------------

        ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

        # Only keep movies that have been rated
        movies = movies[movies["movie_id"].isin(ratings["movie_id"])].copy()

        movie_ft = self._movie_features(movies, ratings)
        user_ft = self._user_features(movies, ratings)
        del movies, ratings

        logger.info("Saving features data ...")
        output_dir = self.data_dir() / "features_data_module"
        output_dir.mkdir(exist_ok=True)

        self.movie_features_path = output_dir / "movie_features.parquet"
        self.user_features_path = output_dir / "user_features.parquet"
        movie_ft.to_parquet(self.movie_features_path, index=False)
        user_ft.to_parquet(self.user_features_path, index=False)

    def setup(self, stage: Optional[str] = None):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        # TODO: Implement
        raise NotImplementedError("Setup is not implemented for FeaturesDataModule.")
