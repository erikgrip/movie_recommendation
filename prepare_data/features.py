"""Feature engineering functions for the movie lens dataset."""

from pathlib import Path
from typing import Optional

import polars as pl
from sentence_transformers import SentenceTransformer

INPUT_DIR = Path("data/clean")
OUTPUT_DIR = Path("data/features")

INPUT_MOVIE_DATA_PATH = INPUT_DIR / "movies.parquet"
INPUT_RATING_DATA_PATH = INPUT_DIR / "ratings.parquet"

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


def text_embedding_polars(movies: pl.DataFrame, dim: Optional[int] = None):
    texts = movies["title"].to_list()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    if dim:
        embeddings = [emb[:dim] for emb in embeddings]
    return pl.DataFrame({"movie_id": movies["movie_id"], "title_embedding": embeddings})


def genre_dummies_polars(movie_data: pl.DataFrame) -> pl.DataFrame:
    """Creates dummy variables for genres in the movie data."""
    present_genres = movie_data["genres"].explode().unique().to_list()
    return (
        movie_data
        # Add empty integer columns for all genres
        .explode("genres")
        .to_dummies("genres")
        .rename({f"genres_{gen}": f"is_{gen}" for gen in GENRES})
        .with_columns(
            [
                pl.lit(0, dtype=pl.Int8).alias(f"is_{gen}")
                for gen in GENRES
                if gen not in present_genres
            ]
        )
        .drop("genres_null")
        .select(["movie_id"] + [f"is_{gen}" for gen in GENRES])
    )


def user_genre_avg_ratings_polars(
    rating_data: pl.DataFrame, movie_genre_dummies: pl.DataFrame
) -> pl.LazyFrame:
    initial_rating = 3.0
    genre_columns = [f"is_{genre}" for genre in GENRES]

    # Merge ratings with genre dummies and convert to LazyFrame
    df = rating_data.join(movie_genre_dummies, on="movie_id", how="left").lazy()

    # Sort data by user_id and timestamp
    df = df.sort(["user_id", "timestamp"])

    # Compute cumulative averages directly without intermediate columns
    avg_rating_exprs = [
        (
            (
                (pl.col(gen) * pl.col("rating")).cum_sum().over("user_id")
                / pl.col(gen).cum_sum().over("user_id")
            )
            .shift(1)
            .fill_null(initial_rating)
            .alias(f"avg_rating_{gen.replace('is_', '')}")
        )
        for gen in genre_columns
    ]
    df = df.with_columns(avg_rating_exprs)

    # Select required columns and collect the result
    result = df.select(
        ["user_id", "timestamp"] + [f"avg_rating_{genre}" for genre in GENRES]
    )

    return result


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Calculating features...")
    ratings = pl.read_parquet(INPUT_RATING_DATA_PATH)
    movies = pl.read_parquet(INPUT_MOVIE_DATA_PATH)

    ## Movie title embeddings
    # print("Calculating movie title embeddings...")
    # title_embeddings = text_embedding_polars(movies)
    # title_embeddings.write_parquet(OUTPUT_DIR / "movie_title_embeddings.parquet")

    # Movie genre dummies
    print("Calculating movie genre dummies...")
    movie_genres = genre_dummies_polars(movies)
    movie_genres.write_parquet(OUTPUT_DIR / "movie_genre_dummies.parquet")

    print("Calculating user genre average ratings...")
    user_genre_avg = user_genre_avg_ratings_polars(ratings, movie_genres)
    user_genre_avg.show_graph()
    user_genre_avg.sink_parquet(OUTPUT_DIR / "user_genre_avg_ratings.parquet")
