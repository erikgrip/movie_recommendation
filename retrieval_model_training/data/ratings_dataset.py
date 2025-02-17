""" Pytorch dataset for the Movie Lens ratings data. """

import torch


class RatingsDataset(torch.utils.data.Dataset):
    """
    A dataset with just the rating user and movie IDs and target.
    """

    keys: set = {"user_label", "movie_label", "user_id", "movie_id", "target"}

    def __init__(self, data: dict[str, list]):
        """Initializes the MovieLensDataset.

        Args:
            data (dict[str, list]):
            {
                "user_label": List[int],
                "movie_label": List[int],
                "user_id": List[int],
                "movie_id": List[int],
                "target": List[float],
            }
        """
        super().__init__()
        self.data = data
        self._validate_data()
        self._length = len(self.data["user_label"])

    def _validate_data(self):
        """
        Validates data to ensure it is the same length and has the correct types.
        """
        if not self.data.keys() == self.keys:
            raise ValueError(
                f"Data must have keys {self.keys}, but got {self.data.keys()}"
            )

        for key, values in self.data.items():
            if len(values) != len(self.data["user_label"]):
                raise ValueError("All data must be the same length")
            if key in {"user_label", "movie_label", "user_id", "movie_id"}:
                if not all(isinstance(v, int) for v in values):
                    raise ValueError(f"{key} must be a list of integers.")
            elif key == "target":
                if not all(isinstance(v, float) for v in values):
                    raise ValueError(f"{key} must be a list of floats.")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.
        """
        return {
            "user_label": torch.tensor(self.data["user_label"][idx], dtype=torch.long),
            "movie_label": torch.tensor(
                self.data["movie_label"][idx], dtype=torch.long
            ),
            "user_id": torch.tensor(self.data["user_id"][idx], dtype=torch.long),
            "movie_id": torch.tensor(self.data["movie_id"][idx], dtype=torch.long),
            "target": torch.tensor(self.data["target"][idx], dtype=torch.float),
        }
