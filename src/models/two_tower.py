""" PyTorch model for Two-Tower recommendation with user and movie features."""

from argparse import ArgumentParser
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.prepare_data.features import GENRES


class UserTower(nn.Module):
    """User tower that processes user features."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.genre_avgs_layer = nn.Sequential(
            nn.Linear(len(GENRES), embedding_dim),
            nn.ReLU(),
        )

    def forward(self, user_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the user tower.
        Args:
            user_features: Tensor of shape (batch_size, num_genres).
        Returns:
            A tensor of shape (batch_size, embedding_dim) representing user embeddings.
        """
        user_embedding = self.genre_avgs_layer(user_features)
        return F.normalize(user_embedding, p=2, dim=1)  # L2 normalization


class MovieTower(nn.Module):
    """Movie tower that processes movie features."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        # TODO: Add movie ID hash
        self.features = [
            "genres",  # dummies
            "release_year",
            "title_embedding",  # pre computed
        ]
        if embedding_dim % len(self.features) != 0:
            raise ValueError(
                f"embedding_dim must be divisible by the number of features: {len(self.features)}"
            )
        self.feature_embedding_dim = int(embedding_dim / len(self.features))

        self.genres_layer = nn.Sequential(
            nn.Linear(len(GENRES), self.feature_embedding_dim),
            nn.ReLU(),
        )
        self.release_year_layer = nn.Sequential(
            nn.Linear(1, self.feature_embedding_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        title_embeddings: torch.Tensor,
        release_year: torch.Tensor,
        genres: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the movie tower.
        Args:
            title_embeddings: Tensor of shape (batch_size, title_embedding_dim).
            release_year: Tensor of shape (batch_size, 1).
            genres: Tensor of shape (batch_size, num_genres).
        Returns:
            A tensor of shape (batch_size, embedding_dim) representing movie embeddings.
        """
        genre_embedding = self.genres_layer(genres)
        year_embedding = self.release_year_layer(release_year.unsqueeze(1))

        concat_embedding = torch.cat(
            [title_embeddings, genre_embedding, year_embedding], dim=1
        )

        return F.normalize(concat_embedding, p=2, dim=1)


class TwoTower(nn.Module):
    """Two-tower recommendation model combining user and movie features."""

    def __init__(self, args: Optional[Dict] = None, **kwargs):
        super().__init__()

        self.args = args or {}
        self.embedding_dim = self.args.get("embedding_dim", 240)

        self.user_tower = UserTower(self.embedding_dim)
        self.movie_tower = MovieTower(self.embedding_dim)

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--embedding_dim",
            type=int,
            default=240,
            help="Dimension of the user and movie embeddings.",
        )
        return parser

    def forward(
        self,
        user_genre_avg: torch.Tensor,
        title_embedding: torch.Tensor,
        release_year: torch.Tensor,
        genres: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the two-tower model.

        Args:
            user_genre_avg: Tensor of shape (batch_size, num_genres).
            title_embedding: Tensor of shape (batch_size, title_embedding_dim).
            release_year: Tensor of shape (batch_size, 1).
            genres: Tensor of shape (batch_size, num_genres).

        Returns:
            A tensor of shape (batch_size,) representing the predicted ratings.
        """
        user_embedding = self.user_tower(user_genre_avg)
        movie_embedding = self.movie_tower(title_embedding, release_year, genres)

        return torch.sum(user_embedding * movie_embedding, dim=1)
