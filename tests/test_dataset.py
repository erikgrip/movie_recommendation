""" Test the dataset module. """

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
    with patch("zipfile.ZipFile", MagicMock(return_value=mock_zipfile)):
        with open(mock_data_path, "rb") as file:
            mock_zipfile.open = MagicMock(return_value=file)
            yield mock_zipfile


@pytest.fixture(name="mock_data")
def fixture_mock_data():
    """Mock some data for the dataset."""
    users = [1, 2, 3]
    movies = [1, 2, 3]
    ratings = [5.0, 4.0, 3.0]
    yield (users, movies, ratings)


def test_movie_lens_dataset(mock_data):
    """Test the MovieLensDataset class."""
    dataset = MovieLensDataset(mock_data[0], mock_data[1], mock_data[2])
    assert len(dataset) == 3
    assert dataset[0] == {
        "users": torch.tensor(1, dtype=torch.long),
        "movies": torch.tensor(1, dtype=torch.long),
        "ratings": torch.tensor(5.0, dtype=torch.float),
    }


def test_movie_lens_dataset_uneven_data_length():
    """Test the MovieLensDataset class with invalid data."""
    with pytest.raises(ValueError):
        MovieLensDataset([1, 2], [1, 2, 3], [5.0, 4.0, 3.0])
    with pytest.raises(ValueError):
        MovieLensDataset([1, 2, 3], [1, 2, 3], [5.0, 4.0])
    with pytest.raises(ValueError):
        MovieLensDataset([1, 2, 3], [1, 2, 3], [5.0, 4.0, 3.0, 2.0])


def test_movie_lens_dataset_invalid_data_type():
    """Test the MovieLensDataset class with invalid data types."""
    with pytest.raises(ValueError):
        MovieLensDataset(["1", "2", "3"], [1, 2, 3], [5.0, 4.0, 3.0])
    with pytest.raises(ValueError):
        MovieLensDataset([1, 2, 3], [1, "2", 3], [5.0, 4.0, 3.0])
    with pytest.raises(ValueError):
        MovieLensDataset([1, 2, 3], [1, 2, 3], [5.0, None, 3.0])
