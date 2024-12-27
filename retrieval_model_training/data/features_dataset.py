""" Pytorch dataset for the Movie Lens ratings data. """

import numpy as np
import pandas as pd
import torch


class FeaturesDataset(torch.utils.data.Dataset):
    """
    A dataset with the rating user and movie features.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        movie_title_embedding: np.ndarray,
        movie_genres: np.ndarray,
        movie_release_year: np.ndarray,
        user_genre_avg: np.ndarray,
        target: np.ndarray,
    ):
        """Initializes the dataset."""
        super().__init__()
        self.movie_title_embedding = movie_title_embedding
        self.movie_genres = movie_genres
        self.movie_release_year = movie_release_year
        self.user_genre_avg = user_genre_avg
        self.target = target

    @classmethod
    def from_pandas(
        cls,
        user_features: pd.DataFrame,
        movie_features: pd.DataFrame,
        target: pd.Series,
    ):
        """Creates a dataset from pandas dataframes."""
        return cls(
            movie_title_embedding=movie_features["title_embedding"].to_numpy(),
            movie_release_year=movie_features["year"].to_numpy(),
            movie_genres=movie_features[
                [m for m in movie_features.columns if m.startswith("is_")]
            ].to_numpy(),
            user_genre_avg=user_features.to_numpy(),
            target=target.to_numpy(),
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.target)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.
        """
        mov_title = self.movie_title_embedding[idx]
        mov_genres = self.movie_genres[idx]  # Collection of dummy variables
        mov_release_year = self.movie_release_year[idx]
        user_genre_avg = self.user_genre_avg[idx]  # Collection of dummy variables

        return {
            "title_embedding": torch.tensor(mov_title, dtype=torch.float32),
            "genres": torch.tensor(mov_genres, dtype=torch.float32),
            "release_year": torch.tensor(mov_release_year, dtype=torch.float32),
            "user_genre_avg": torch.tensor(user_genre_avg, dtype=torch.float32),
            "target": torch.tensor(self.target[idx], dtype=torch.float32),
        }
