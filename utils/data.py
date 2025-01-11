from typing import Tuple

import pandas as pd

COL_RENAME = {"movieId": "movie_id", "userId": "user_id", "rating": "target"}


def time_split_data(
    df: pd.DataFrame, test_frac: float, val_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into train, validation, and test sets."""
    df = df.sort_values(by="timestamp", ascending=True)
    n_rows = len(df)
    val_size = int(n_rows * val_frac)
    test_size = int(n_rows * test_frac)

    train_data = df[: -(test_size + val_size)]
    val_data = df[-(test_size + val_size) : -test_size]
    test_data = df[-test_size:]

    if len(train_data) + len(val_data) + len(test_data) != n_rows:
        raise ValueError("Data split error")

    return train_data, val_data, test_data
