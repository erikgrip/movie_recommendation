"""Cleans the raw data and saves the cleaned data to disk."""

import sys
from pathlib import Path

import pandas as pd

from prepare_data.config import (
    CLEAN_DIR,
    CLEAN_MOVIE_DATA_PATH,
    CLEAN_RATING_DATA_PATH,
    EXTRACTED_MOVIE_DATA_PATH,
    EXTRACTED_RATING_DATA_PATH,
)


def extract_movie_release_year(titles: pd.Series) -> pd.Series:
    """Extracts year from the movie title and returns it as a float.

    For example, for the title "Toy Story (1995)", this function will return 1995.0.
    The output will have null values for movies where the year could not be extracted.
    """
    return titles.str.extract(r"\((\d{4})\)").astype("float")[0]


def clean_movie_titles(titles: pd.Series) -> pd.Series:
    """Cleans the movie titles by removing the year and trailing whitespace."""
    return titles.str.replace(r"\((\d{4})\)", "", regex=True).str.strip()


def impute_missing_year(
    movie_data: pd.DataFrame, rating_data: pd.DataFrame
) -> pd.DataFrame:
    """Imputes missing years in the movies DataFrame using the year of the first rating.

    Converts the year column to an integer type.
    """
    year_first_rated = (
        rating_data.sort_values("timestamp")
        .drop_duplicates("movie_id", keep="first")
        .set_index("movie_id")["timestamp"]
        .apply(lambda x: x.year)
    )
    mask = movie_data["year"].isna()
    movie_data.loc[mask, "year"] = movie_data.loc[mask, "movie_id"].map(
        year_first_rated
    )
    movie_data["year"] = movie_data["year"].astype("int")
    return movie_data


def clean_genres(genres: pd.Series) -> pd.Series:
    """Cleans the genres column."""
    return (
        genres.str.replace("-", "_")
        .str.lower()
        .str.split("|")
        .apply(lambda x: [gen for gen in x if gen != "(no genres listed)"])
    )


if __name__ == "__main__":
    if CLEAN_MOVIE_DATA_PATH.exists() and CLEAN_RATING_DATA_PATH.exists():
        print("Cleaned data already exists.")
        sys.exit(0)
    if not Path(CLEAN_DIR).exists():
        Path(CLEAN_DIR).mkdir(parents=True)

    COL_RENAME = {"movieId": "movie_id", "userId": "user_id"}
    movies = pd.read_csv(EXTRACTED_MOVIE_DATA_PATH).rename(columns=COL_RENAME)
    ratings = pd.read_csv(EXTRACTED_RATING_DATA_PATH).rename(columns=COL_RENAME)

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    # Drop movies that have not been rated
    movies = movies[movies["movie_id"].isin(ratings["movie_id"])]

    movies["release_year"] = extract_movie_release_year(movies["title"])
    movies = impute_missing_year(movies, ratings)
    movies["title"] = clean_movie_titles(movies["title"])
    movies["genres"] = clean_genres(movies["genres"])

    movies.to_parquet(CLEAN_MOVIE_DATA_PATH, index=False)
    ratings.to_parquet(CLEAN_RATING_DATA_PATH, index=False)
