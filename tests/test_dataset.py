""" Test the dataset module. """

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data.ratings_dataset import MovieLensDataset


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
    return {
        "user_label": [0, 1, 2],
        "movie_label": [0, 1, 2],
        "user_id": [1, 2, 3],
        "movie_id": [1, 2, 3],
        "rating": [5.0, 4.0, 3.0],
    }


def test_movie_lens_dataset(mock_data):
    """Test the MovieLensDataset class."""
    dataset = MovieLensDataset(mock_data)
    assert len(dataset) == 3
    assert dataset[0] == {
        "user_label": torch.tensor(0, dtype=torch.long),
        "movie_label": torch.tensor(0, dtype=torch.long),
        "user_id": torch.tensor(1, dtype=torch.long),
        "movie_id": torch.tensor(1, dtype=torch.long),
        "rating": torch.tensor(5.0, dtype=torch.float),
    }


def test_movie_lens_dataset_keys(mock_data):
    """Test the MovieLensDataset class with invalid keys."""
    mock_data["extra_key"] = [1, 2, 3]
    with pytest.raises(ValueError):
        MovieLensDataset(mock_data)

    mock_data.pop("extra_key")
    mock_data.pop("user_label")
    with pytest.raises(ValueError):
        MovieLensDataset(mock_data)


def test_movie_lens_dataset_uneven_data_length(mock_data):
    """Test the MovieLensDataset class with invalid data."""
    mock_data["user_id"].append(4)
    with pytest.raises(ValueError):
        MovieLensDataset(mock_data)


def test_movie_lens_dataset_invalid_data_type(mock_data):
    """Test the MovieLensDataset class with invalid data types."""
    mock_data["user_id"][0] = "1"  # String instead of int
    with pytest.raises(ValueError):
        MovieLensDataset(mock_data)

    mock_data["user_id"][0] = 1
    mock_data["rating"][0] = 5  # Int instead of float
    with pytest.raises(ValueError):
        MovieLensDataset(mock_data)
