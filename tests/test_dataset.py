from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data.dataset import MovieLensDataset


@pytest.fixture(name="mock_open")
def fixture_mock_open():
    """Mock the zipfile.ZipFile.open to use the test fixture file."""
    mock_data_path = "tests/fixtures/ratings.csv"
    with patch("src.data.dataset.zipfile.ZipFile.open") as mock_open:
        mock_open.return_value = MagicMock()
        mock_open.return_value.__enter__.return_value = open(mock_data_path, "rb")
        yield mock_open


def test_movie_lens_dataset(mock_open):  # pylint: disable=unused-argument
    """Test the MovieLensDataset class."""
    dataset = MovieLensDataset()
    assert len(dataset) == 3
    assert dataset[0] == {
        "users": torch.tensor(1, dtype=torch.long),
        "movies": torch.tensor(1, dtype=torch.long),
        "ratings": torch.tensor(5.0, dtype=torch.float),
    }
