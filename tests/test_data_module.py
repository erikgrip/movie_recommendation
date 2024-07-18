"""Test suite for the MovieLensDataModule class."""

from unittest.mock import patch

import pandas as pd  # type: ignore
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.data_module import MovieLensDataModule
from tests.mocking import (  # pylint: disable=unused-import
    fixture_data_module, fixture_mock_zip)


@pytest.fixture(name="mock_csv", autouse=True)
def fixture_mock_csv():
    """Fixture to mock pd.read_csv to return a test fixture dataframe."""
    mock_df = pd.read_csv("tests/fixtures/ratings.csv")
    with patch("src.data.data_module.pd.read_csv") as mock_csv:
        mock_csv.return_value = mock_df
        yield mock_csv


@pytest.mark.parametrize(
    "args,expected_batch_size,expected_num_workers,expected_test_frac",
    [
        ({}, 32, 0, 0.2),
        ({"batch_size": 64, "num_workers": 4, "test_frac": 0.1}, 64, 4, 0.1),
    ],
)
def test_init(args, expected_batch_size, expected_num_workers, expected_test_frac):
    """Test the initialization of MovieLensDataModule."""
    data_module = MovieLensDataModule(args)
    mock_data_dir = data_module.data_dirname()
    assert data_module.zip_path == str(mock_data_dir / "ml-latest.zip")
    assert data_module.data_path == str(mock_data_dir / "ratings.csv")
    assert data_module.test_frac == expected_test_frac
    assert data_module.batch_size == expected_batch_size
    assert data_module.num_workers == expected_num_workers


@pytest.mark.parametrize("test_frac", [-0.1, 0.04, 0.96])
def test_init_invalid_test_fraction(test_frac):
    """Test initialization with invalid test fraction values."""
    with pytest.raises(ValueError):
        MovieLensDataModule(args={"test_frac": test_frac})


def test_num_labels_before_setup_raises_error():
    """Test that getting number of labels raise an error before setup."""
    data_module = MovieLensDataModule()
    data_module.prepare_data()
    with pytest.raises(ValueError):
        data_module.num_user_labels()
    with pytest.raises(ValueError):
        data_module.num_movie_labels()


def test_num_labels_after_prepare_data():
    """Test num_user_labels and num_movie_labels after setup."""
    data_module = MovieLensDataModule()
    data_module.prepare_data()
    data_module.setup()
    assert data_module.num_user_labels() == 2
    assert data_module.num_movie_labels() == 2


def test_prepare_data():
    """Test the prepare_data method."""
    data_module = MovieLensDataModule()
    data_module.prepare_data()
    with open(data_module.data_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    assert data[0].strip() == "userId,movieId,rating,timestamp"
    assert len(data) == 4


def test_prepare_data_already_exctracted():
    """Test the prepare_data method when the data is already extracted."""
    data_module = MovieLensDataModule()
    with (
        patch("src.data.data_module.Path.exists") as mock_exists,
        patch("src.data.data_module.zipfile.ZipFile") as mock_zip,
    ):
        mock_exists.return_value = True
        data_module.prepare_data()
        assert not mock_zip.called


@pytest.mark.parametrize(
    "frac,expected_train_len,expected_test_len",
    [(0.1, 3, 0), (0.33, 2, 1), (0.67, 1, 2), (0.9, 0, 3)],
)
def test_setup(frac, expected_train_len, expected_test_len):
    """Test the setup method for different test fractions."""
    data_module = MovieLensDataModule(args={"test_frac": frac})

    data_module.setup("fit")
    train_len = len(data_module.train_dataset)
    assert train_len == expected_train_len
    assert data_module.test_dataset is None
    if train_len > 0:
        # Oldest rating should be at the end
        assert data_module.train_dataset[-1] == {
            "movies": torch.tensor(0),
            "ratings": torch.tensor(4.0),
            "users": torch.tensor(1),
        }

    data_module.setup("test")
    assert len(data_module.test_dataset) == expected_test_len


def test_train_dataloader():
    """Test the train_dataloader method."""
    data_module = MovieLensDataModule(args={"test_frac": 0.33})
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()

    assert isinstance(train_dataloader, DataLoader)
    assert len(train_dataloader.dataset) == 2


def test_test_dataloader():
    """Test the test_dataloader method."""
    data_module = MovieLensDataModule(args={"test_frac": 0.33})
    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()

    assert isinstance(test_dataloader, DataLoader)
    assert len(test_dataloader.dataset) == 1
