"""Feature engineering functions for the movie lens dataset."""

from pathlib import Path
from tempfile import TemporaryDirectory as tempdir
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer

GENRES = [
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "imax",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]

INPUT_DIR = Path("data/clean")
OUTPUT_DIR = Path("data/features")

INPUT_MOVIE_DATA_PATH = INPUT_DIR / "movies.parquet"
INPUT_RATING_DATA_PATH = INPUT_DIR / "ratings.parquet"

# SentenceTransformer model name to use for text embeddings.
# NOTE: If updated - make sure to use a Matryoshka model, i.e. a model
# that produces embeddings that can be truncated to a smaller dimension.
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-xsmall-v1"


def text_embedding(texts: list[str], dim: Optional[int] = None) -> list:
    """Calculates the sentence embeddings for the given text using the SentenceTransformer model."""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    return [emb[:dim] for emb in embeddings] if dim else embeddings


def genre_dummies(movie_data: pd.DataFrame) -> pd.DataFrame:
    """Extracts dummy features from movie dataframe `genres` column.

    Returns a new dataframe with the movie_id and dummy columns for each genre.
    """
    dummies = (
        movie_data["genres"]
        .str.get_dummies(sep="|")
        .drop(columns="(no genres listed)", errors="ignore")
        .rename(columns=lambda x: "is_" + x.lower().replace("-", "_"))
    )
    return pd.concat([movie_data["movie_id"], dummies], axis=1)


def user_genre_avg_ratings(
    rating_data: pd.DataFrame, movie_genre_dummies: pd.DataFrame
) -> pd.DataFrame:
    """Calculates the average rating a user has given to each genre at each point in time.

    Has 3 as initial value for each genre.

    Example:
    Input ratings:
    user_id  movie_id  target   timestamp
    1        1         5        2021-01-01 10:00:00
    1        2         4        2021-01-20 13:00:00
    1        3         3        2021-02-05 15:00:00
    1        4         2        2021-02-10 09:00:00
    2        3         4        2021-01-05 10:00:00
    2        1         4        2021-03-02 10:00:00

    Input movie_genres:
    movie_id  action comedy ...
    1         1      0
    2         0      1
    3         1      0
    4         0      0

    Returns:
    user_id  timestamp            avg_rating_action avg_rating_comedy ...
    1        2021-01-01 10:00:00  3.0               3.0
    1        2021-01-20 13:00:00  5.0               3.0
    1        2021-02-05 15:00:00  5.0               4.0
    1        2021-02-10 09:00:00  4.0               4.0
    2        2021-01-05 10:00:00  3.0               3.0
    2        2021-03-02 10:00:00  3.0               4.0
    """
    initial_rating = 3
    calc_columns = [f"avg_rating_{col}" for col in GENRES]

    df = rating_data.merge(movie_genre_dummies, on="movie_id")

    # Add genre even if it wasn't watched
    for genre in GENRES:
        if genre not in df.columns:
            df[genre] = 0

    def add_base_columns(cols: list, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Map aggregated columns to the original dataframe."""
        return pd.concat([df[cols], dataframe], axis=1)

    # Calculate the average rating for each genre at each point in time
    avg = (
        add_base_columns(["user_id"], df[GENRES].multiply(df["rating"], axis=0))
        .groupby("user_id")
        .cumsum()
    ).div(df.groupby("user_id")[GENRES].cumsum(), axis=0)
    avg.columns = pd.Index(calc_columns)

    # Shift the average ratings by one to avoid data leakage
    result = (
        add_base_columns(["user_id"], avg)
        .groupby(["user_id"])[calc_columns]
        .shift(1)
        .fillna(initial_rating)
    )

    return add_base_columns(["user_id", "timestamp"], result)


def calculate_features(
    rating_data: pd.DataFrame, movie_data: pd.DataFrame, embedding_dim: int = 256
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates user features from the ratings and movies dataframes."""
    movie_genre_dummies = genre_dummies(movie_data)

    users = user_genre_avg_ratings(rating_data, movie_genre_dummies)

    movie_data["title_embedding"] = text_embedding(movie_data["title"], embedding_dim)

    movie_data = (
        movie_data.drop(columns=["title", "genres"])
        .merge(movie_genre_dummies, on="movie_id")
        .rename(columns={k: "is_" + k for k in GENRES})
    )
    return movie_data, users


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_parquet(INPUT_RATING_DATA_PATH)
    movies = pd.read_parquet(INPUT_MOVIE_DATA_PATH)

    # Movie title embeddings
    # pd.DataFrame(
    #    {
    #        "movie_id": movies["movie_id"].to_list(),
    #        "title_embedding": text_embedding(movies["title"].to_list()),
    #    }
    # ).to_parquet(OUTPUT_DIR / "movie_title_embeddings.parquet")

    # Movie genre dummies
    movie_genres = genre_dummies(movies)
    movie_genres.to_parquet(OUTPUT_DIR / "movie_genre_dummies.parquet")

    # User genre average ratings
    user_genre_avg_ratings(ratings, movie_genres).to_parquet(
        OUTPUT_DIR / "user_genre_avg_ratings.parquet"
    )
