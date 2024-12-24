""" Pytorch dataset for the Movie Lens ratings data. """

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class FeaturesDataset(torch.utils.data.Dataset):
    """
    A dataset with the rating user and movie features.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        movie_titles: np.ndarray,
        movie_genres: np.ndarray,
        movie_release_years: np.ndarray,
        user_genre_avgs: np.ndarray,
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
    ):
        """Initializes the dataset."""
        super().__init__()
        self.movie_titles = movie_titles
        self.movie_genres = movie_genres
        self.movie_release_years = movie_release_years
        self.user_genre_avgs = user_genre_avgs
        self.labels = labels
        self.tokenizer = tokenizer

    @classmethod
    def from_pandas(
        cls,
        user_features: pd.DataFrame,
        movie_features: pd.DataFrame,
        labels: pd.Series,
        tokenizer: AutoTokenizer,
    ):
        """Creates a dataset from pandas dataframes."""
        return cls(
            movie_titles=movie_features["title"].to_numpy(),
            movie_genres=movie_features["genres"].to_numpy(),
            movie_release_years=movie_features["year"].to_numpy(),
            user_genre_avgs=user_features.to_numpy(),
            labels=labels.to_numpy(),
            tokenizer=tokenizer,
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.
        """
        raise NotImplementedError
