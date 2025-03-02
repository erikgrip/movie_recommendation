""" Recommendation model with user and movie embeddings 

Copied from https://pureai.substack.com/p/recommender-systems-with-pytorch
"""

from argparse import ArgumentParser
from typing import Dict, Optional

import torch
from torch import nn

EMBEDDING_SIZE = 256
HIDDEN_DIM = 256
DROPOUT_RATE = 0.2


class NeuralCollaborativeFilteringModel(
    nn.Module
):  # pylint: disable=too-many-instance-attributes
    """
    PyTorch model for neural collaborative filtering with user and movie embeddings.
    This combines matrix factorization with neural networks to capture non-linear relationships.
    """

    def __init__(self, num_users: int, num_movies: int, args: Optional[Dict] = None):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        args = args or {}
        self.embedding_size = args.get("embedding_size", EMBEDDING_SIZE)
        self.hidden_dim = args.get("hidden_dim", HIDDEN_DIM)
        self.dropout_rate = args.get("dropout_rate", DROPOUT_RATE)

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
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Activation function
        self.relu = nn.ReLU()

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> ArgumentParser:
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--embedding_size",
            type=int,
            default=EMBEDDING_SIZE,
            help=f"Number of units in the embedding layer (default: {EMBEDDING_SIZE})",
        )
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=HIDDEN_DIM,
            help=f"Number of units in the hidden layer (default: {HIDDEN_DIM})",
        )
        parser.add_argument(
            "--dropout_rate",
            type=float,
            default=DROPOUT_RATE,
            help=f"Dropout rate (default: {DROPOUT_RATE})",
        )
        return parser

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
