""" PyTorch model for Two-Tower recommendation with user and movie features."""

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn


class UserTower(nn.Module):
    """User tower that processes user features."""

    def __init__(self, user_feature_dim: int, embedding_dim: int):
        super().__init__()
        self.user_fc = nn.Sequential(
            nn.Linear(user_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

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

    def __init__(self, movie_feature_dim: int, embedding_dim: int):
        super().__init__()
        self.title_embedding_dim = 768  # Assumes BERT produces embeddings of this size

        # Fully connected layers for movie features
        self.movie_fc = nn.Sequential(
            nn.Linear(movie_feature_dim + self.title_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

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

    def __init__(
        self,
        user_feature_dim: int,
        movie_feature_dim: int,
        embedding_dim: int,
        **kwargs
    ):
        super().__init__()
        self.user_tower = UserTower(user_feature_dim, embedding_dim)
        self.movie_tower = MovieTower(movie_feature_dim, embedding_dim)

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        """Add model-specific arguments to the parser."""
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
