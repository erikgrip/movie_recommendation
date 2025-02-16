import sys
from pathlib import Path

import pandas as pd

from utils.data import COL_RENAME
from utils.log import logger

THRESHOLD = 4  # Threshold for binary classification

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

rating_data_path = DATA_DIR / "extracted" / "ratings.csv"
movie_data_path = DATA_DIR / "extracted" / "movies.csv"

output_dir = DATA_DIR / "ranking_features"
output_dir.mkdir(parents=True, exist_ok=True)

movie_features_path = output_dir / "movie_features.parquet"
user_features_path = output_dir / "user_features.parquet"


# Load data
if __name__ == "__main__":

    if movie_features_path.exists() and user_features_path.exists():
        logger.info("Features data already exists. Skipping preparation.")
        sys.exit(0)
    elif rating_data_path.exists() and movie_data_path.exists():
        logger.info("Ratings and movie data data already exists.")
    else:
        raise FileNotFoundError(
            f"Data not found at {rating_data_path} and {movie_data_path}. "
            "Please run the data preparation script first."
        )

    movies = pd.read_csv(movie_data_path).rename(columns=COL_RENAME)
    ratings = pd.read_csv(rating_data_path).rename(columns=COL_RENAME)

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    # Calculate features
    movie_ft, user_ft = pd.DataFrame(), pd.DataFrame()

    df = pd.merge_asof(
        ratings.sort_values("timestamp"),
        user_ft.sort_values("timestamp"),
        by="user_id",
        on="timestamp",
        direction="backward",  # Use the latest snapshot before the interaction
    ).merge(movie_ft, on="movie_id")

    print("Merged data:")
    print(df.head())

    df["target"] = df["target"] >= THRESHOLD

    print("Data with target:")
    df.info()
