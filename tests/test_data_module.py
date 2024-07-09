"""Test suite for the MovieLensDataModule class."""

import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from torch.utils.data import DataLoader

from src.data.data_module import MovieLensDataModule


@pytest.fixture(name="mock_zip", autouse=True)
def fixture_mock_zip():
    """Fixture to mock zipfile.ZipFile.open with a test fixture file."""
    mock_data_path = "tests/fixtures/ratings.csv"
    mock_zipfile = MagicMock()
    mock_zipfile.__enter__.return_value = mock_zipfile
    with patch(
        "src.data.data_module.zipfile.ZipFile", MagicMock(return_value=mock_zipfile)
    ):
        with open(mock_data_path, "rb") as file:
            mock_zipfile.open = MagicMock(return_value=file)
            yield mock_zipfile


@pytest.fixture(name="mock_csv", autouse=True)
def fixture_mock_csv():
    """Fixture to mock pd.read_csv to return a test fixture dataframe."""
    mock_df = pd.read_csv("tests/fixtures/ratings.csv")
    with patch("src.data.data_module.pd.read_csv") as mock_csv:
        mock_csv.return_value = mock_df
        yield mock_csv


@pytest.fixture(name="mock_output_path", autouse=True)
def fixture_mock_output_path():
    """Fixture to create a temporary directory for the output path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield f"{tmp_dir}/ratings.csv"


def test_init():
    """Test the initialization of MovieLensDataModule."""
    data_module = MovieLensDataModule()

    assert data_module.data_path == "data/ml-latest.zip"
    assert data_module.output_path == "data/ratings.csv"
    assert data_module.test_frac == 0.1


@pytest.mark.parametrize("test_frac", [-0.1, 0.04, 0.96])
def test_init_invalid_test_fraction(test_frac):
    """Test initialization with invalid test fraction values."""
    with pytest.raises(ValueError):
        MovieLensDataModule(test_frac)


def test_prepare_data(mock_output_path):
    """Test the prepare_data method."""
    with patch(
        "src.data.data_module.MovieLensDataModule.output_path", mock_output_path
    ):
        data_module = MovieLensDataModule()
        data_module.prepare_data()
        with open(mock_output_path, "r", encoding="utf-8") as file:
            data = file.readlines()
        assert data[0].strip() == "userId,movieId,rating,timestamp"
        assert len(data) == 4


@pytest.mark.parametrize(
    "frac,expected_train_len,expected_test_len",
    [(0.1, 3, 0), (0.33, 2, 1), (0.67, 1, 2), (0.9, 0, 3)],
)
def test_setup(frac, expected_train_len, expected_test_len):
    """Test the setup method for different test fractions."""
    data_module = MovieLensDataModule(test_frac=frac)
    data_module.setup("fit")

    assert len(data_module.train_dataset) == expected_train_len
    assert data_module.test_dataset is None

    data_module.setup("test")
    assert len(data_module.test_dataset) == expected_test_len


def test_train_dataloader():
    """Test the train_dataloader method."""
    data_module = MovieLensDataModule(test_frac=0.33)
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()

    assert isinstance(train_dataloader, DataLoader)
    assert len(train_dataloader.dataset) == 2


def test_test_dataloader():
    """Test the test_dataloader method."""
    data_module = MovieLensDataModule(test_frac=0.33)
    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()

    assert isinstance(test_dataloader, DataLoader)
    assert len(test_dataloader.dataset) == 1
