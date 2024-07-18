""" Recommendation model with user and movie embeddings 

Copied from https://pureai.substack.com/p/recommender-systems-with-pytorch
"""

from typing import Dict, Optional

import torch
from torch import nn


class RecommendationModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    A PyTorch model for recommendation with user and movie embeddings.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_size: int = 256,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        args: Optional[Dict] = None,
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_size
        )
        self.movie_embedding = nn.Embedding(
            num_embeddings=self.num_movies, embedding_dim=self.embedding_size
        )

        # Hidden layers
        self.fc1 = nn.Linear(2 * self.embedding_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        # Embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)

        # Concatenate user and movie embeddings
        combined = torch.cat([user_embedded, movie_embedded], dim=1)

        # Pass through hidden layers with ReLU activation and dropout
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)

        return output
