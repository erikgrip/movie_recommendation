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
        self._validate_data()

    def _validate_data(self):
        """
        Validates data to ensure it is the same length and has the correct types.
        """
        if not len(self.users) == len(self.movies) == len(self.ratings):
            raise ValueError("users, movies, and ratings must be the same length")
        if not all(isinstance(u, int) for u in self.users):
            raise ValueError("users must be a list of integers")
        if not all(isinstance(m, int) for m in self.movies):
            raise ValueError("movies must be a list of integers")
        if not all(isinstance(r, float) for r in self.ratings):
            raise ValueError("ratings must be a list of floats")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.users)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.
        """
        return {
            "users": torch.tensor(self.users[idx], dtype=torch.long),
            "movies": torch.tensor(self.movies[idx], dtype=torch.long),
            "ratings": torch.tensor(self.ratings[idx], dtype=torch.float),
        }
