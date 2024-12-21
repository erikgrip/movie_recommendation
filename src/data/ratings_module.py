""" PyTorch Lightning data module for the MovieLens ratings data. """

import warnings
from typing import Dict, Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder  # type: ignore

from src.data.base_module import BaseDataModule
from src.data.ratings_dataset import RatingsDataset
from src.prepare_data.download_dataset import download_and_extract_data
from src.utils.log import logger

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
            pd.read_csv(self.rating_data_path)
            # TODO: Remove this line to get predictions working for new movies
            .sort_values(by="timestamp", ascending=False)[dtypes.keys()]
            .astype(dtypes)
            .rename(columns={"userId": "user_id", "movieId": "movie_id"})
        )

        # ---------------------
        # TODO: Drop sampling down
        logger.info("Downsampling data ...")
        sample_users = df["user_id"].unique()[:100]
        df = df[df["user_id"].isin(sample_users)]
        # ---------------------

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
