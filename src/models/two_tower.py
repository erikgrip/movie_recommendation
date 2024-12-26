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
        self.user_fc = nn.Embedding(len(GENRES), embedding_dim)

    def forward(self, user_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the user tower.
        Args:
            user_features: Tensor of shape (batch_size, user_feature_dim).
        Returns:
            A tensor of shape (batch_size, embedding_dim) representing user embeddings.
        """
        user_embedding = self.user_fc(user_features)
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

        self.genres_fc = nn.Embedding(len(GENRES), self.feature_embedding_dim)
        self.release_year_fc = nn.Embedding(2024 - 1800, self.feature_embedding_dim)

    def forward(self, title_embeddings: torch.Tensor, other_features: torch.Tensor):
        """
        Forward pass for the movie tower.
        Args:
            title_embeddings: Tensor of shape (batch_size, title_embedding_dim).
            other_features: Tensor of shape (batch_size, movie_feature_dim).
        Returns:
            A tensor of shape (batch_size, embedding_dim) representing movie embeddings.
        """
        combined_features = torch.cat([title_embeddings, other_features], dim=1)
        movie_embedding = self.movie_fc(combined_features)
        return F.normalize(movie_embedding, p=2, dim=1)  # L2 normalization


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
        user_features: torch.Tensor,
        title_embeddings: torch.Tensor,
        movie_features: torch.Tensor,
    ):
        """
        Forward pass for the Two-Tower model.
        Args:
            user_features: Tensor of shape (batch_size, user_feature_dim).
            title_embeddings: Tensor of shape (batch_size, title_embedding_dim).
            movie_features: Tensor of shape (batch_size, movie_feature_dim).
        Returns:
            user_embedding: Tensor of shape (batch_size, embedding_dim).
            movie_embedding: Tensor of shape (batch_size, embedding_dim).
        """
        user_embedding = self.user_tower(user_features)
        movie_embedding = self.movie_tower(title_embeddings, movie_features)
        return user_embedding, movie_embedding

    def predict(
        self,
        user_features: torch.Tensor,
        title_embeddings: torch.Tensor,
        movie_features: torch.Tensor,
    ):
        """
        Compute the similarity score between user and movie embeddings.
        Args:
            user_features: Tensor of shape (batch_size, user_feature_dim).
            title_embeddings: Tensor of shape (batch_size, title_embedding_dim).
            movie_features: Tensor of shape (batch_size, movie_feature_dim).
        Returns:
            Tensor of shape (batch_size,) representing similarity scores.
        """
        user_embedding, movie_embedding = self.forward(
            user_features, title_embeddings, movie_features
        )
        return torch.sum(
            user_embedding * movie_embedding, dim=1
        )  # Dot product similarity
