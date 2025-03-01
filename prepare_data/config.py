"""Configuration for the data preparation pipeline."""

from pathlib import Path

DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"

DATA_DIR = Path("data")

# Directory structure
ZIP_SAVE_DIR = DATA_DIR / "raw"
EXTRACT_DIR = DATA_DIR / "extracted"
CLEAN_DIR = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# File paths
ZIP_SAVE_PATH = ZIP_SAVE_DIR / "ml-latest.zip"

# Extracted files mapping
EXTRACTED_FILES = {
    "ml-latest/ratings.csv": EXTRACT_DIR / "ratings.csv",
    "ml-latest/movies.csv": EXTRACT_DIR / "movies.csv",
}

# Extracted files paths
EXTRACTED_MOVIE_DATA_PATH = EXTRACT_DIR / "movies.csv"
EXTRACTED_RATING_DATA_PATH = EXTRACT_DIR / "ratings.csv"

# Clean data paths
CLEAN_MOVIE_DATA_PATH = CLEAN_DIR / "movies.parquet"
CLEAN_RATING_DATA_PATH = CLEAN_DIR / "ratings.parquet"

# Feature paths
FEATURE_MOVIE_TITLE_EMBEDDINGS_PATH = FEATURES_DIR / "movie_title_embeddings.parquet"
FEATURE_MOVIE_GENRE_DUMMIES_PATH = FEATURES_DIR / "movie_genre_dummies.parquet"
FEATURE_USER_GENRE_AVG_RATINGS_PATH = FEATURES_DIR / "user_genre_avg_ratings.parquet"
