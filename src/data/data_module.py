""" PyTorch Lightning data module for the MovieLens ratings data. """

import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import DataLoader

from src.data.dataset import MovieLensDataset
from src.data.utils import (
    FILES_TO_EXTRACT,
    ZIP_SAVE_PATH,
    download_zip,
    extract_files,
    extracted_files_exist,
    zip_exists,
)
from src.utils.log import logger

warnings.filterwarnings("ignore", category=FutureWarning)

BATCH_SIZE = 32
NUM_WORKERS = 0
TEST_FRAC = 0.2


class MovieLensDataModule(pl.LightningDataModule):
    """Lightning data module for the MovieLens ratings data."""

    def __init__(self, args: Optional[Dict] = None):
        super().__init__()
        args = args or {}
        self.batch_size = args.get("batch_size", BATCH_SIZE)
        self.num_workers = args.get("num_workers", NUM_WORKERS)
        self.test_frac = args.get("test_frac", TEST_FRAC)
        self._validate_test_frac()

        self.train_dataset: MovieLensDataset
        self.test_dataset: MovieLensDataset
        self.user_label_encoder: LabelEncoder = LabelEncoder()
        self.movie_label_encoder: LabelEncoder = LabelEncoder()

    def _validate_test_frac(self):
        if not 0.05 < self.test_frac < 0.95:
            raise ValueError("test_frac must be between 0.05 and 0.95")

    @classmethod
    def data_dirname(cls) -> Path:
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
            "--test_frac",
            type=float,
            default=TEST_FRAC,
            help=f"Fraction of data to use for testing (default: {TEST_FRAC})",
        )
        return parser

    @property
    def data_path(self) -> str:
        """Return the path to the ratings data."""
        return str(
            self.data_dirname()
            / FILES_TO_EXTRACT["ml-latest/ratings.csv"].rsplit("/", maxsplit=1)[-1]
        )

    @property
    def movie_path(self) -> str:
        """Return the path to the movies metadata."""
        return str(
            self.data_dirname()
            / FILES_TO_EXTRACT["ml-latest/movies.csv"].rsplit("/", maxsplit=1)[-1]
        )

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
        if extracted_files_exist([self.data_path, self.movie_path]):
            logger.info("Data is already downloaded and extracted.")
            return

        if not zip_exists():
            logger.info("Downloading MovieLens data to %s ...", ZIP_SAVE_PATH)
            download_zip()
        logger.info("Extracting data to %s ...", " ".join(FILES_TO_EXTRACT.values()))
        extract_files(self.data_dirname())

    def setup(self, stage: str = ""):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
        df = (
            pd.read_csv(self.data_path)
            .sort_values(by="timestamp", ascending=False)[dtypes.keys()]
            .astype(dtypes)
            .rename(columns={"userId": "user_id", "movieId": "movie_id"})
        )

        df["user_label"] = self.user_label_encoder.fit_transform(df["user_id"])
        df["movie_label"] = self.movie_label_encoder.fit_transform(df["movie_id"])

        test_size = round(len(df) * self.test_frac)

        def to_input_data(df: pd.DataFrame) -> Dict[str, list]:
            return {str(k): list(v) for k, v in df.to_dict(orient="list").items()}

        if stage == "fit":
            self.train_dataset = MovieLensDataset(to_input_data(df.iloc[test_size:]))

        if stage in ("test", "predict"):
            self.test_dataset = MovieLensDataset(to_input_data(df.iloc[:test_size]))

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

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
