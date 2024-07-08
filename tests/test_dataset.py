from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data.dataset import MovieLensDataset


@pytest.fixture(name="mock_data_file")
def fixture_mock_open():
    """Mock the zipfile.ZipFile.open to use the test fixture file."""
    mock_data_path = "tests/fixtures/ratings.csv"
    mock_zipfile = MagicMock()
    mock_zipfile.__enter__.return_value = mock_zipfile
    mock_zipfile.open = MagicMock(return_value=open(mock_data_path, "rb"))
    with patch("zipfile.ZipFile", MagicMock(return_value=mock_zipfile)):
        yield mock_zipfile


def test_movie_lens_dataset(mock_data_file):  # pylint: disable=unused-argument
    """Test the MovieLensDataset class."""
    dataset = MovieLensDataset()
    assert len(dataset) == 3
    assert dataset[0] == {
        "users": torch.tensor(1, dtype=torch.long),
        "movies": torch.tensor(1, dtype=torch.long),
        "ratings": torch.tensor(5.0, dtype=torch.float),
    }
