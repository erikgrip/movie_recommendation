import csv
from io import TextIOWrapper
import zipfile
import torch

DATA_PATH = "data/ml-latest.zip"
RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


class MovieLensDataset(torch.utils.data.Dataset):
    """
    The Movie Lens Dataset class. This class prepares the dataset for training and validation.
    """

    def __init__(self):
        """
        Initializes the dataset object with user, movie, and rating data.
        """
        data = self._read_ratings()
        self.users = [int(user) for user in data["userId"]]
        self.movies = [int(movie) for movie in data["movieId"]]
        self.ratings = [float(rating) for rating in data["rating"]]

    @staticmethod
    def _read_ratings() -> dict:
        """
        Loads the ratings data from the zip file.
        """
        data = []
        with zipfile.ZipFile(DATA_PATH) as archive:
            with archive.open("ratings.csv") as file:
                reader = csv.reader(TextIOWrapper(file, "utf-8"))
                for row in reader:
                    data.append(row)
                if data[0] != RATING_COLUMNS:
                    raise ValueError(
                        f"Invalid rating file headers. Expected {RATING_COLUMNS}, got {data[0]}"
                    )

        return dict(zip(RATING_COLUMNS, zip(*data[1:])))

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
