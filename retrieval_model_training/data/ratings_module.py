""" PyTorch Lightning data module for the MovieLens ratings data. """

import warnings
from typing import Dict, Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder  # type: ignore

from prepare_data.download import download_and_extract_data
from retrieval_model_training.data.base_module import BaseDataModule
from retrieval_model_training.data.ratings_dataset import RatingsDataset
from utils.data import COL_RENAME, time_split_data
from utils.log import logger

warnings.filterwarnings("ignore", category=FutureWarning)


class RatingsDataModule(BaseDataModule):
    """Lightning data module for the MovieLens ratings data."""

    def __init__(self, args: Optional[Dict] = None):
        super().__init__(args)

        self.train_dataset: RatingsDataset
        self.val_dataset: RatingsDataset
        self.test_dataset: RatingsDataset
        self.user_label_encoder: LabelEncoder = LabelEncoder()
        self.movie_label_encoder: LabelEncoder = LabelEncoder()

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

    def prepare_data(self) -> None:
        """Download data and other preparation steps to be done only once."""
        if self.rating_data_path.exists():
            logger.info("Ratings data already exists.")
        else:
            download_and_extract_data()

    def setup(self, stage: Optional[str] = None):
        """Split the data into train and test sets and other setup steps to be done once per GPU."""
        dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
        df = (
            pd.read_csv(self.rating_data_path).astype(dtypes).rename(columns=COL_RENAME)
        )

        df["user_label"] = self.user_label_encoder.fit_transform(df["user_id"])
        df["movie_label"] = self.movie_label_encoder.fit_transform(df["movie_id"])

        train, val, test = time_split_data(
            df, test_frac=self.test_frac, val_frac=self.val_frac
        )

        def to_dict(df: pd.DataFrame) -> Dict[str, list]:
            return {
                str(k): list(v)
                for k, v in df.to_dict(orient="list").items()
                if k != "timestamp"
            }

        if stage == "fit":
            self.train_dataset = RatingsDataset(to_dict(train))
            self.val_dataset = RatingsDataset(to_dict(val))
        if stage in ("test", "predict"):
            self.test_dataset = RatingsDataset(to_dict(test))
