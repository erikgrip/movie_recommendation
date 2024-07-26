"""Test suite for the MovieLensDataModule class."""

# pylint: disable=unused-import

from unittest.mock import patch

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.data_module import MovieLensDataModule
from tests.mocking import fixture_data_module


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
    assert data_module.num_user_labels() == 100
    assert data_module.num_movie_labels() == 97


def test_prepare_data():
    """Test the prepare_data method."""
    data_module = MovieLensDataModule()
    data_module.prepare_data()
    with open(data_module.data_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    assert data[0].strip() == "userId,movieId,rating,timestamp"
    assert len(data) == 101


def test_prepare_data_already_exctracted():
    """Test the prepare_data method when the data is already extracted."""
    data_module = MovieLensDataModule()
    with (
        patch("src.data.data_module.Path.exists") as mock_exists,
        patch("src.data.data_module.download_zip") as mock_zip,
        patch("src.data.data_module.extract_files") as mock_extract,
    ):
        mock_exists.return_value = True
        data_module.prepare_data()
        assert not mock_zip.called and not mock_extract.called


@pytest.mark.parametrize(
    "frac,expected_train_len,expected_test_len",
    [(0.1, 90, 10), (0.33, 67, 33), (0.67, 33, 67), (0.9, 10, 90)],
)
def test_setup(frac, expected_train_len, expected_test_len):
    """Test the setup method for different test fractions."""
    data_module = MovieLensDataModule(args={"test_frac": frac})
    data_module.prepare_data()

    data_module.setup("fit")
    train_len = len(data_module.train_dataset)
    assert train_len == expected_train_len

    print(data_module.movie_label_encoder.classes_)

    if train_len > 0:
        # Oldest rating should be at the end
        assert data_module.train_dataset[-1] == {
            "movie_label": torch.tensor(12),
            "user_label": torch.tensor(16),
            "movie_id": torch.tensor(485),
            "rating": torch.tensor(3.0),
            "user_id": torch.tensor(72565),
        }

    data_module.setup("test")
    assert len(data_module.test_dataset) == expected_test_len


@pytest.mark.parametrize(
    "args,expected_batch_size,expected_num_workers,expected_len",
    [
        ({}, 32, 0, 80),
        ({"batch_size": 64, "num_workers": 4, "test_frac": 0.1}, 64, 4, 90),
    ],
)
def test_train_dataloader(
    args, expected_batch_size, expected_num_workers, expected_len
):
    """Test the train_dataloader method."""
    data_module = MovieLensDataModule(args)
    data_module.prepare_data()
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()

    assert isinstance(train_dataloader, DataLoader)
    assert len(train_dataloader.dataset) == expected_len
    assert train_dataloader.batch_size == expected_batch_size
    assert train_dataloader.num_workers == expected_num_workers


@pytest.mark.parametrize(
    "args,expected_batch_size,expected_num_workers,expected_len",
    [
        ({}, 32, 0, 20),
        ({"batch_size": 64, "num_workers": 4, "test_frac": 0.1}, 64, 4, 10),
    ],
)
def test_test_dataloader(args, expected_batch_size, expected_num_workers, expected_len):
    """Test the test_dataloader method."""
    data_module = MovieLensDataModule(args)
    data_module.prepare_data()
    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()

    assert isinstance(test_dataloader, DataLoader)
    assert len(test_dataloader.dataset) == expected_len
    assert test_dataloader.batch_size == expected_batch_size
    assert test_dataloader.num_workers == expected_num_workers
