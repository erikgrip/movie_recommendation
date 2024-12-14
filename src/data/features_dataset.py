""" Pytorch dataset for the Movie Lens ratings data. """

import torch


class FeaturesDataset(torch.utils.data.Dataset):
    """
    A dataset with the rating user and movie features.
    """

    def __init__(self, data: dict[str, list]):
        """
        Initializes the MovieLensDataset.

        Args:
            data (dict[str, list]):
            {
                "user_id": List[int],
                "user_genre_fractions": List[List[float]],
                "user_genre_avg_ratings": List[List[float]],
                "movie_title": List[str],
                "movie_year": List[int],
                "movie_genres": List[List[str]],
                "rating": List[float],
            }
        """

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        pass

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.
        """
        pass
