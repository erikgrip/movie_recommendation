""" Pytorch dataset for the Movie Lens ratings data. """

import numpy as np
import pandas as pd
import torch


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
    ):
        """Initializes the dataset."""
        super().__init__()
        self.movie_titles = movie_titles
        self.movie_genres = movie_genres
        self.movie_release_years = movie_release_years
        self.user_genre_avgs = user_genre_avgs
        self.labels = labels

    @classmethod
    def from_pandas(
        cls,
        user_features: pd.DataFrame,
        movie_features: pd.DataFrame,
        labels: pd.Series,
    ):
        """Creates a dataset from pandas dataframes."""
        return cls(
            # TODO: Fix naming
            movie_titles=movie_features["title_embedding"].to_numpy(),
            movie_release_years=movie_features["year"].to_numpy(),
            movie_genres=movie_features[
                [m for m in movie_features.columns if m.startswith("is_")]
            ].to_numpy(),
            user_genre_avgs=user_features.to_numpy(),
            labels=labels.to_numpy(),
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
        mov_title = self.movie_titles[idx]
        mov_genres = self.movie_genres[idx]  # Collection of dummy variables
        mov_release_year = self.movie_release_years[idx]
        user_pref = self.user_genre_avgs[idx]  # Collection of dummy variables

        return {
            "title_embedding": torch.tensor(mov_title, dtype=torch.float32),
            "genres": torch.tensor(mov_genres, dtype=torch.float32),
            "release_year": torch.tensor(mov_release_year, dtype=torch.float32),
            "user_pref": torch.tensor(user_pref, dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
