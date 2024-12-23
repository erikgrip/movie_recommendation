# pylint: disable=missing-function-docstring
""" Tests for the features module """

from unittest.mock import patch

import pandas as pd

from src.prepare_data.features import (
    genre_dummies,
    movie_genres_to_list,
    user_genre_avg_ratings,
)


def test_movie_genres_to_list():
    movies = pd.Series(["Action", "Action|Thriller|Comedy", "(no genres listed)"])

    expected_output = pd.Series([["Action"], ["Action", "Thriller", "Comedy"], []])
    output = movie_genres_to_list(movies)

    pd.testing.assert_series_equal(output, expected_output)


def test_genre_dummies():
    movies = pd.DataFrame({"movie_id": [1, 2], "genres": ["Action", "Action|Thriller"]})

    expected_output = pd.DataFrame(
        {"movie_id": [1, 2], "action": [1, 1], "thriller": [0, 1]}
    )
    output = genre_dummies(movies)

    pd.testing.assert_frame_equal(output, expected_output)


def test_genre_dummies_filter():
    movies = pd.DataFrame(
        {"movie_id": [1, 2], "genres": ["Action", "(no genres listed)"]}
    )

    expected_output = pd.DataFrame({"movie_id": [1, 2], "action": [1, 0]})
    output = genre_dummies(movies)

    pd.testing.assert_frame_equal(output, expected_output)


def test_genre_dummies_rename():
    movies = pd.DataFrame({"movie_id": [1], "genres": ["Action|Sci-Fi"]})

    expected_output = pd.DataFrame({"movie_id": [1], "action": [1], "sci_fi": [1]})
    output = genre_dummies(movies)

    pd.testing.assert_frame_equal(output, expected_output)


def test_user_genre_avg_ratings():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2],
            "movie_id": [1, 2, 3, 4, 3, 1],
            "rating": [5, 4, 3, 2, 4, 4],
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01 10:00:00",
                    "2021-01-20 13:00:00",
                    "2021-02-05 15:00:00",
                    "2021-02-10 09:00:00",
                    "2021-01-05 10:00:00",
                    "2021-03-02 10:00:00",
                ]
            ),
        }
    )

    movie_genre_dummies = pd.DataFrame(
        [
            {"movie_id": 1, "action": 1, "comedy": 0},
            {"movie_id": 2, "action": 0, "comedy": 1},
            {"movie_id": 3, "action": 1, "comedy": 0},
            {"movie_id": 4, "action": 0, "comedy": 0},
        ]
    )

    expected_output = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2],
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01 10:00:00",
                    "2021-01-20 13:00:00",
                    "2021-02-05 15:00:00",
                    "2021-02-10 09:00:00",
                    "2021-01-05 10:00:00",
                    "2021-03-02 10:00:00",
                ]
            ),
            "avg_rating_action": [3.0, 5.0, 5.0, 4.0, 3.0, 4.0],
            "avg_rating_comedy": [3.0, 3.0, 4.0, 4.0, 3.0, 3.0],
            "avg_rating_some_other_genre": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # Not rated
        }
    )

    with patch(
        "src.prepare_data.features.GENRES", ["action", "comedy", "some_other_genre"]
    ):
        output = user_genre_avg_ratings(ratings, movie_genre_dummies)

    pd.testing.assert_frame_equal(output, expected_output)
