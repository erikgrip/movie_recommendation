import pandas as pd


def movie_genres_to_list(genres: pd.Series) -> pd.Series:
    """Converts the genres column to a list of genres."""
    return genres.replace("(no genres listed)", "").str.split("|")


def extract_movie_release_year(titles: pd.Series) -> pd.Series:
    """Extracts year from the movie title and returns it as a float.

    For example, for the title "Toy Story (1995)", this function will return 1995.0.
    The output will have null values for movies where the year could not be extracted.
    """
    return titles.str.extract(r"\((\d{4})\)").astype("float")


def clean_movie_titles(titles: pd.Series) -> pd.Series:
    """Cleans the movie titles by removing the year and trailing whitespace."""
    return titles.str.replace(r"\((\d{4})\)", "", regex=True).str.strip()


def impute_missing_year(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing years in the movies DataFrame using the year of the first rating.

    Converts the year column to an integer type.
    """
    year_first_rated = (
        ratings.sort_values("datetime")
        .drop_duplicates("movie_id", keep="first")
        .set_index("movie_id")["datetime"]
        .apply(lambda x: x.year)
    )
    mask = movies["year"].isna()
    movies.loc[mask, "year"] = movies.loc[mask, "movie_id"].map(year_first_rated)
    movies["year"] = movies["year"].astype("int")
    return movies
