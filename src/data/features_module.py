""" PyTorch Lightning data module for the MovieLens ratings data. """

import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import DataLoader

from src.data.features_dataset import FeaturesDataset
from src.utils import features
from src.utils.log import logger

warnings.filterwarnings("ignore", category=FutureWarning)

BATCH_SIZE = 32
NUM_WORKERS = 0
VAL_FRAC = 0.1
TEST_FRAC = 0.1


class FeaturesDataModule(
    pl.LightningDataModule
):  # pylint: disable=too-many-instance-attributes
    """Lightning data module for the MovieLens ratings data."""

    def __init__(self, args: Optional[Dict] = None):
        super().__init__()
        args = args or {}
        self.batch_size = args.get("batch_size", BATCH_SIZE)
        self.num_workers = args.get("num_workers", NUM_WORKERS)
        self.val_frac = args.get("val_frac", VAL_FRAC)
        self.test_frac = args.get("test_frac", TEST_FRAC)
        self._validate_data_fractions()

        self.movie_features_path: Path
        self.user_features_path: Path

        self.train_dataset: FeaturesDataset
        self.val_dataset: FeaturesDataset
        self.test_dataset: FeaturesDataset

        # multilingual bert tokenizer
        # self.tokenizer = transformers.BertTokenizer.from_pretrained(
        #     "bert-base-multilingual-cased"
        # )
        self.user_label_encoder: LabelEncoder = LabelEncoder()
        self.movie_label_encoder: LabelEncoder = LabelEncoder()

    def _validate_data_fractions(self):
        """Ensure that the data fractions are valid."""
        if not 0 < self.val_frac < 1.0:
            raise ValueError("Validation fraction must be between 0 and 1.")
        if not 0 < self.test_frac < 1.0:
            raise ValueError("Test fraction must be between 0 and 1.")
        if not 0 < self.val_frac + self.test_frac < 1.0:
            raise ValueError("Validation and test fractions must sum to less than 1.0.")

    @classmethod
    def data_path(cls) -> Path:
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

    def prepare_data(self):
        """Download data and other preparation steps to be done only once."""
        features_dir = self.data_path() / "features_data_module"
        os.makedirs(features_dir, exist_ok=True)

        if (features_dir / "movie_features.parquet").exists() and (
            features_dir / "user_features.parquet"
        ).exists():
            logger.info("Features data already exists. Skipping preparation.")
            return

        # Load data
        col_rename = {"movieId": "movie_id", "userId": "user_id"}
        movies = pd.read_csv(self.data_path() / "extracted/movies.csv").rename(
            columns=col_rename
        )
        ratings = pd.read_csv(self.data_path() / "extracted/ratings.csv").rename(
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
        output_dir = self.data_path() / "features_data_module"
        output_dir.mkdir(exist_ok=True)

        self.movie_features_path = output_dir / "movie_features.parquet"
        self.user_features_path = output_dir / "user_features.parquet"
        movie_ft.to_parquet(self.movie_features_path, index=False)
        user_ft.to_parquet(self.user_features_path, index=False)

    def setup(self, stage: str = ""):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
        df = (
            pd.read_csv(self.data_path() / "extracted/ratings.csv")
            # TODO: Remove this line to get predictions working for new movies
            .sort_values(by="timestamp", ascending=False)[dtypes.keys()]
            .astype(dtypes)
            .rename(columns={"userId": "user_id", "movieId": "movie_id"})
        )

        df["user_label"] = self.user_label_encoder.fit_transform(df["user_id"])
        df["movie_label"] = self.movie_label_encoder.fit_transform(df["movie_id"])

        test_split = round(len(df) * self.test_frac)
        val_split = round(len(df) * (self.test_frac + self.val_frac))

        def to_input_data(df: pd.DataFrame) -> Dict[str, list]:
            return {str(k): list(v) for k, v in df.to_dict(orient="list").items()}

        if stage == "fit":
            self.train_dataset = FeaturesDataset(to_input_data(df.iloc[val_split:]))
            self.val_dataset = FeaturesDataset(
                to_input_data(df.iloc[test_split:val_split])
            )

        if stage in ("test", "predict"):
            self.test_dataset = FeaturesDataset(to_input_data(df.iloc[:test_split]))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
