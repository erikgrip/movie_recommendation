# pylint: disable=missing-function-docstring
""" Tests for the features module """

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from prepare_data.features import (
    calculate_user_genre_avg_ratings,
    genre_dummies,
    text_embedding,
)


def test_genre_dummies():
    movies = pd.DataFrame({"movie_id": [1, 2], "genres": ["Action", "Action|Thriller"]})

    expected_output = pd.DataFrame(
        {"movie_id": [1, 2], "is_action": [1, 1], "is_thriller": [0, 1]}
    )
    output = genre_dummies(movies)

    pd.testing.assert_frame_equal(output, expected_output)


def test_genre_dummies_filter():
    movies = pd.DataFrame(
        {"movie_id": [1, 2], "genres": ["Action", "(no genres listed)"]}
    )

    expected_output = pd.DataFrame({"movie_id": [1, 2], "is_action": [1, 0]})
    output = genre_dummies(movies)

    pd.testing.assert_frame_equal(output, expected_output)


def test_genre_dummies_rename():
    movies = pd.DataFrame({"movie_id": [1], "genres": ["Action|Sci-Fi"]})

    expected_output = pd.DataFrame(
        {"movie_id": [1], "is_action": [1], "is_sci_fi": [1]}
    )
    output = genre_dummies(movies)

    pd.testing.assert_frame_equal(output, expected_output)


@pytest.mark.parametrize(
    "dim,expected_output",
    [
        (1, [[0.1], [0.4], [0.7]]),
        (2, [[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]]),
        (3, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        (4, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
    ],
)
def test_text_embedding(dim, expected_output):
    titles = ["title1", "title2", "title3"]

    with patch("prepare_data.features.SentenceTransformer") as mock_st:
        mock_st.return_value.encode.return_value = np.array(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.4, 0.5, 0.6]),
                np.array([0.7, 0.8, 0.9]),
            ]
        )

        output = text_embedding(titles, dim=dim)

    np.testing.assert_array_equal(output, expected_output)


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
        "prepare_data.features.GENRES", ["action", "comedy", "some_other_genre"]
    ):
        output = calculate_user_genre_avg_ratings(ratings, movie_genre_dummies)

    pd.testing.assert_frame_equal(output, expected_output)
