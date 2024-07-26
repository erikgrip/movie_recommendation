""" Pytorch dataset for the Movie Lens ratings data. """

import torch


class MovieLensDataset(torch.utils.data.Dataset):
    """
    The Movie Lens Dataset class.
    """

    keys: set = {"user_label", "movie_label", "user_id", "movie_id", "rating"}

    def __init__(self, data: dict[str, list]):
        """Initializes the MovieLensDataset.

        Args:
            data (dict[str, list]):
            {
                "user_label": List[int],
                "movie_label": List[int],
                "user_id": List[int],
                "movie_id": List[int],
                "rating": List[float],
            }
        """
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
            elif key == "rating":
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
            "rating": torch.tensor(self.data["rating"][idx], dtype=torch.float),
        }
