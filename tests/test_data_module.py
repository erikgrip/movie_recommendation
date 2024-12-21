"""Test suite for the MovieLensDataModule class."""

# pylint: disable=unused-import

from unittest.mock import patch

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.ratings_module import RatingsDataModule
from tests.mocking import fixture_ratings_data_module


@pytest.mark.parametrize(
    "args,expected_batch_size,expected_num_workers,expected_val_frac,expected_test_frac",
    [
        ({}, 32, 0, 0.1, 0.1),
        ({"batch_size": 64, "num_workers": 4, "test_frac": 0.2}, 64, 4, 0.1, 0.2),
    ],
)
def test_init(
    args,
    expected_batch_size,
    expected_num_workers,
    expected_val_frac,
    expected_test_frac,
):
    """Test the initialization of RatingsDataModule."""
    data_module = RatingsDataModule(args)
    mock_data_dir = data_module.data_dir()
    assert data_module.rating_data_path == mock_data_dir / "extracted" / "ratings.csv"
    assert data_module.val_frac == expected_val_frac
    assert data_module.test_frac == expected_test_frac
    assert data_module.batch_size == expected_batch_size
    assert data_module.num_workers == expected_num_workers


@pytest.mark.parametrize(
    "val_frac,test_frac", [(0.1, 0.9), (0.9, 0.1), (0.0, 0.5), (0.5, 1.0), (-0.1, 0.5)]
)
def test_init_invalid_val_and_test_fraction(test_frac, val_frac):
    """Test initialization with invalid val and test fractions."""
    with pytest.raises(ValueError):
        RatingsDataModule(args={"test_frac": test_frac, "val_frac": val_frac})


def test_num_labels_before_setup_raises_error():
    """Test that getting number of labels raise an error before setup."""
    data_module = RatingsDataModule()
    data_module.prepare_data()
    with pytest.raises(ValueError):
        data_module.num_user_labels()
    with pytest.raises(ValueError):
        data_module.num_movie_labels()


def test_num_labels_after_prepare_data():
    """Test num_user_labels and num_movie_labels after setup."""
    data_module = RatingsDataModule()
    data_module.prepare_data()
    data_module.setup()
    assert data_module.num_user_labels() == 10
    assert data_module.num_movie_labels() == 97


def test_prepare_data():
    """Test the prepare_data method."""
    data_module = RatingsDataModule()
    data_module.prepare_data()
    with open(data_module.rating_data_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    assert data[0].strip() == "userId,movieId,rating,timestamp"
    assert len(data) == 101


def test_prepare_data_already_exctracted():
    """Test the prepare_data method when the data is already extracted."""
    data_module = RatingsDataModule()
    with (
        patch(
            "src.data.ratings_module.RatingsDataModule.rating_data_path"
        ) as mock_data,
        patch("src.data.ratings_module.download_and_extract_data") as mock_get_data,
    ):
        mock_data.exists.return_value = True
        data_module.prepare_data()
        mock_get_data.assert_not_called()


@pytest.mark.parametrize(
    "val_frac,test_frac,expected_train_len,expected_val_len,expected_test_len",
    [(0.1, 0.1, 80, 10, 10), (0.2, 0.1, 70, 20, 10), (0.1, 0.2, 70, 10, 20)],
)
def test_setup(
    val_frac, test_frac, expected_train_len, expected_val_len, expected_test_len
):
    """Test the setup method for different test fractions."""
    data_module = RatingsDataModule(args={"val_frac": val_frac, "test_frac": test_frac})
    data_module.prepare_data()

    data_module.setup("fit")
    train_len = len(data_module.train_dataset)
    val_len = len(data_module.val_dataset)

    assert train_len == expected_train_len
    assert val_len == expected_val_len
    # Oldest rating should be at the end
    assert data_module.train_dataset[-1] == {
        "movie_label": torch.tensor(16),
        "user_label": torch.tensor(3),
        "movie_id": torch.tensor(590),
        "rating": torch.tensor(4.0),
        "user_id": torch.tensor(128783),
    }

    data_module.setup("test")
    assert len(data_module.test_dataset) == expected_test_len


@pytest.mark.parametrize(
    "args,expected_batch_size,expected_num_workers,expected_len",
    [
        ({}, 32, 0, 80),
        (
            {"batch_size": 64, "num_workers": 4, "val_frac": 0.2, "test_frac": 0.2},
            64,
            4,
            60,
        ),
    ],
)
def test_train_dataloader(
    args, expected_batch_size, expected_num_workers, expected_len
):
    """Test the train_dataloader method."""
    data_module = RatingsDataModule(args)
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
        ({}, 32, 0, 10),
        ({"batch_size": 64, "num_workers": 4, "val_frac": 0.2}, 64, 4, 20),
    ],
)
def test_val_dataloader(args, expected_batch_size, expected_num_workers, expected_len):
    """Test the val_dataloader method."""
    data_module = RatingsDataModule(args)
    data_module.prepare_data()
    data_module.setup("fit")
    val_dataloader = data_module.val_dataloader()

    assert isinstance(val_dataloader, DataLoader)
    assert len(val_dataloader.dataset) == expected_len
    assert val_dataloader.batch_size == expected_batch_size
    assert val_dataloader.num_workers == expected_num_workers


@pytest.mark.parametrize(
    "args,expected_batch_size,expected_num_workers,expected_len",
    [
        ({}, 32, 0, 10),
        ({"batch_size": 64, "num_workers": 4, "test_frac": 0.2}, 64, 4, 20),
    ],
)
def test_test_dataloader(args, expected_batch_size, expected_num_workers, expected_len):
    """Test the test_dataloader method."""
    data_module = RatingsDataModule(args)
    data_module.prepare_data()
    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()

    assert isinstance(test_dataloader, DataLoader)
    assert len(test_dataloader.dataset) == expected_len
    assert test_dataloader.batch_size == expected_batch_size
    assert test_dataloader.num_workers == expected_num_workers
