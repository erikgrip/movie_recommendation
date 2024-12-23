"""Feature engineering functions for the movie lens dataset."""

from typing import Tuple

import pandas as pd

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


def movie_genres_to_list(genres: pd.Series) -> pd.Series:
    """Converts the genres column to a list of genres."""
    return genres.str.split("|").apply(
        lambda x: x if x != ["(no genres listed)"] else []
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


def impute_missing_year(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing years in the movies DataFrame using the year of the first rating.

    Converts the year column to an integer type.
    """
    year_first_rated = (
        ratings.sort_values("timestamp")
        .drop_duplicates("movie_id", keep="first")
        .set_index("movie_id")["timestamp"]
        .apply(lambda x: x.year)
    )
    mask = movies["year"].isna()
    movies.loc[mask, "year"] = movies.loc[mask, "movie_id"].map(year_first_rated)
    movies["year"] = movies["year"].astype("int")
    return movies


def genre_dummies(movies: pd.DataFrame) -> pd.DataFrame:
    """Extracts dummy features from movie dataframe `genres` column.

    Returns a new dataframe with the movie_id and dummy columns for each genre.
    """
    dummies = (
        movies["genres"]
        .str.get_dummies(sep="|")
        .drop(columns="(no genres listed)", errors="ignore")
        .rename(columns=lambda x: x.lower().replace("-", "_"))
    )
    return pd.concat([movies["movie_id"], dummies], axis=1)


def user_genre_avg_ratings(
    ratings: pd.DataFrame, movie_genre_dummies: pd.DataFrame
) -> pd.DataFrame:
    """Calculates the average rating a user has given to each genre at each point in time.

    Has 3 as initial value for each genre.

    Example:
    Input ratings:
    user_id  movie_id  rating   timestamp
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
    user_id  timestamp             avg_rating_action avg_rating_comedy ...
    1        2021-01-01 10:00:00  3.0               3.0
    1        2021-01-20 13:00:00  5.0               3.0
    1        2021-02-05 15:00:00  5.0               4.0
    1        2021-02-10 09:00:00  4.0               4.0
    2        2021-01-05 10:00:00  3.0               3.0
    2        2021-03-02 10:00:00  3.0               4.0
    """
    initial_rating = 3
    calc_columns = [f"avg_rating_{col}" for col in GENRES]

    df = ratings.merge(movie_genre_dummies, on="movie_id")

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
    ratings: pd.DataFrame, movies: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates user features from the ratings and movies dataframes."""
    movie_genre_dummies_df = genre_dummies(movies)

    movies["genres"] = movie_genres_to_list(movies["genres"])
    movies["year"] = extract_movie_release_year(movies["title"])
    movies = impute_missing_year(movies, ratings)
    movies["title"] = clean_movie_titles(movies["title"])

    users = user_genre_avg_ratings(ratings, movie_genre_dummies_df)
    return movies, users
