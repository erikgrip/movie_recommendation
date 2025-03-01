"""Feature engineering functions for the movie lens dataset."""

import tempfile
from glob import glob
from typing import Optional

import polars as pl
from sentence_transformers import SentenceTransformer

from prepare_data.config import (
    CLEAN_MOVIE_DATA_PATH,
    CLEAN_RATING_DATA_PATH,
    FEATURE_MOVIE_GENRE_DUMMIES_PATH,
    FEATURE_MOVIE_TITLE_EMBEDDINGS_PATH,
    FEATURE_USER_GENRE_AVG_RATINGS_PATH,
    FEATURES_DIR,
)
from utils.log import logger

# SentenceTransformer model name to use for text embeddings.
# NOTE: If updated - make sure to use a Matryoshka model, i.e. a model
# that produces embeddings that can be truncated to a smaller dimension.
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-xsmall-v1"

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


def movie_title_embeddings(movie_df: pl.DataFrame, dim: Optional[int] = None):
    """Calculates text embeddings for movie titles using SentenceTransformer."""
    texts = movie_df["title"].to_list()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    if dim:
        embeddings = [emb[:dim] for emb in embeddings]
    return pl.DataFrame(
        {"movie_id": movie_df["movie_id"], "title_embedding": embeddings}
    )


def genre_dummies(movie_data: pl.DataFrame) -> pl.DataFrame:
    """Creates dummy variables for all defined genres."""
    present_genres = movie_data["genres"].explode().drop_nulls().unique().to_list()
    return (
        movie_data.explode("genres")
        .to_dummies("genres")
        .group_by("movie_id", maintain_order=True)
        .agg(pl.sum(*[f"genres_{gen}" for gen in present_genres]))
        .rename({f"genres_{gen}": f"is_{gen}" for gen in GENRES}, strict=False)
        .with_columns(
            [
                pl.lit(0).alias(f"is_{gen}")
                for gen in GENRES
                if gen not in present_genres
            ]
        )
        .drop(["genres_null"], strict=False)
        .select(["movie_id"] + [f"is_{gen}" for gen in GENRES])
        .cast(pl.Int64)
    )


def calculate_user_genre_avg_ratings(
    rating_data: pl.DataFrame, movie_genre_dummies: pl.DataFrame
) -> pl.DataFrame:
    """Calculates the average rating for each genre for each user."""
    initial_rating = 3
    genre_columns = [f"is_{genre}" for genre in GENRES]

    # Merge ratings with genre dummies and convert to LazyFrame
    df = rating_data.join(movie_genre_dummies, on="movie_id", how="left").lazy()

    # Sort data by user_id and timestamp
    df = df.sort(["user_id", "timestamp"])

    # Compute cumulative averages directly without intermediate columns
    avg_rating_exprs = [
        (
            (pl.col(gen) * pl.col("rating")).cum_sum().shift(1).over("user_id")
            / pl.col(gen).cum_sum().shift(1).over("user_id")
        ).alias(f"avg_rating_{gen.replace('is_', '')}")
        for gen in genre_columns
    ]

    return (
        df.with_columns(avg_rating_exprs)
        .select(["user_id", "timestamp"] + [f"avg_rating_{genre}" for genre in GENRES])
        .collect()
        .fill_nan(initial_rating)
        .fill_null(initial_rating)
    )


if __name__ == "__main__":
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Reading data...")
    ratings = pl.read_parquet(CLEAN_RATING_DATA_PATH)
    movies = pl.read_parquet(CLEAN_MOVIE_DATA_PATH)

    logger.info("Calculating movie title embeddings...")
    title_embeddings = movie_title_embeddings(movies)
    title_embeddings.write_parquet(FEATURE_MOVIE_TITLE_EMBEDDINGS_PATH)

    logger.info("Creating movie genre dummies...")
    movie_genres = genre_dummies(movies)
    movie_genres.write_parquet(FEATURE_MOVIE_GENRE_DUMMIES_PATH)

    user_ids = ratings["user_id"].unique().to_list()

    NUM_CHUNKS = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        for chunk in range(1, NUM_CHUNKS + 1):
            if chunk % 10 == 0:
                print(f"Processing chunk {chunk} of {NUM_CHUNKS}...")

            # Split user IDs into chunks
            chunk_size = len(user_ids) // NUM_CHUNKS
            chunk_user_ids = user_ids[(chunk - 1) * chunk_size : chunk * chunk_size]

            chunk_ratings = ratings.filter(pl.col("user_id").is_in(chunk_user_ids))
            chunk_user_genre_avg = calculate_user_genre_avg_ratings(
                chunk_ratings, movie_genres
            )
            chunk_user_genre_avg.write_parquet(
                f"{tmpdir}/user_genre_avg_ratings_{chunk}.parquet"
            )

        user_genre_avg_ratings = pl.concat(
            [pl.scan_parquet(file) for file in glob(f"{tmpdir}/*.parquet")]
        )
        user_genre_avg_ratings.sink_parquet(FEATURE_USER_GENRE_AVG_RATINGS_PATH)
