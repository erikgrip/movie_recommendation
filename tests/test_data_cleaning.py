# pylint: disable=missing-function-docstring
""" Tests for the data cleaning module """

import numpy as np
import pandas as pd

from prepare_data.clean import (
    clean_movie_titles,
    extract_movie_release_year,
    impute_missing_year,
)


def test_extract_movie_release_year():
    titles = pd.Series(["Toy Story (1995)", "Jumanji (1995)", "No Year"])

    expected_output = pd.Series([1995.0, 1995.0, np.nan], name=0)
    output = extract_movie_release_year(titles)

    pd.testing.assert_series_equal(output, expected_output)


def test_impute_missing_year():
    movies = pd.DataFrame({"movie_id": [1, 2, 3], "year": [1950.0, np.nan, np.nan]})

    ratings = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "timestamp": pd.to_datetime(
                ["1951-01-01 10:00:00", "1999-01-20 13:00:00", "2021-02-05 15:00:00"]
            ),
        }
    )

    expected_output = pd.DataFrame(
        {"movie_id": [1, 2, 3], "year": [1950, 1999, 2021]}, dtype=int
    )
    output = impute_missing_year(movies, ratings)

    pd.testing.assert_frame_equal(output, expected_output)


def test_clean_movie_titles():
    titles = pd.Series(["Toy Story (1995)", "Jumanji (1995)", "No Year"])

    expected_output = pd.Series(["Toy Story", "Jumanji", "No Year"])
    output = clean_movie_titles(titles)

    pd.testing.assert_series_equal(output, expected_output)
