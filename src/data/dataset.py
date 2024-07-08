""" Pytorch dataset for the Movie Lens ratings data. """

import torch


class MovieLensDataset(torch.utils.data.Dataset):
    """
    The Movie Lens Dataset class.
    """

    def __init__(self, users: list[int], movies: list[int], ratings: list[float]):
        """
        Initializes the dataset object with user, movie, and rating data.
        """
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.users)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.
        """
        users = self.users[idx]
        movies = self.movies[idx]
        ratings = self.ratings[idx]

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }
