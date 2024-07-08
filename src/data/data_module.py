import zipfile
from io import TextIOWrapper

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset import MovieLensDataset

DATA_DIR = "data"
DATA_PATH = f"{DATA_DIR}/ml-latest.zip"
OUTPUT_PATH = f"{DATA_DIR}/ratings.csv"


class MovieLensDataModule(pl.LightningDataModule):
    """Lightning data module for the MovieLens ratings data."""

    def __init__(self, data_dir: str = DATA_PATH, test_frac: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.test_frac = test_frac
        self.data_train = None
        self.data_test = None

    def prepare_data(self):
        with zipfile.ZipFile(DATA_PATH) as archive:
            with archive.open(DATA_PATH) as file:
                pd.read_csv(TextIOWrapper(file, "utf-8")).to_csv(
                    OUTPUT_PATH, index=False
                )

    def setup(self, stage: str):

        df = pd.read_csv(OUTPUT_PATH).sort_values(by="timestamp", ascending=False)[
            ["userId", "movieId", "rating"]
        ]
        test_size = int(len(df) * self.test_frac)

        if stage == "fit":
            self.data_train = MovieLensDataset(
                *df.iloc[test_size:].to_dict(orient="list").values()
            )

        if stage == "test":
            self.data_test = MovieLensDataset(
                *df.iloc[:test_size].to_dict(orient="list").values()
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=32)
