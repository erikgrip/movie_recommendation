""" PyTorch Lightning data module for the MovieLens ratings data. """

import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import DataLoader

from src.data.ratings_dataset import RatingsDataset

warnings.filterwarnings("ignore", category=FutureWarning)

BATCH_SIZE = 32
NUM_WORKERS = 0
VAL_FRAC = 0.1
TEST_FRAC = 0.1


class RatingsDataModule(
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

        self.train_dataset: RatingsDataset
        self.val_dataset: RatingsDataset
        self.test_dataset: RatingsDataset
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

    @property
    def data_path(self) -> str:
        """Return the path to the ratings data."""
        return str(self.data_dirname() / "extracted/ratings.csv")

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
        if not Path(self.data_path).exists():
            raise FileNotFoundError(
                f"File {self.data_path} not found. Please run `python src/data/download_dataset.py`."
            )

    def setup(self, stage: str = ""):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
        df = (
            pd.read_csv(self.data_path)
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
            self.train_dataset = RatingsDataset(to_input_data(df.iloc[val_split:]))
            self.val_dataset = RatingsDataset(
                to_input_data(df.iloc[test_split:val_split])
            )

        if stage in ("test", "predict"):
            self.test_dataset = RatingsDataset(to_input_data(df.iloc[:test_split]))

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
