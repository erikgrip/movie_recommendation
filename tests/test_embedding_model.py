""" Tests for the FactorizationModel class. """

import pytest
import torch

from src.models.factorization import FactorizationModel


@pytest.fixture(name="model")
def fixture_model():
    """Create an example FactorizationModel."""
    num_users = 100
    num_movies = 200
    args = {
        "embedding_size": 256,
        "hidden_dim": 256,
        "dropout_rate": 0.2,
    }
    return FactorizationModel(num_users=num_users, num_movies=num_movies, args=args)


def test_factorization_model_init(model):
    """Test the initialization of FactorizationModel."""
    assert model.num_users == 100
    assert model.num_movies == 200
    assert model.embedding_size == 256
    assert model.hidden_dim == 256
    assert model.dropout_rate == 0.2


@pytest.mark.parametrize(
    "users, movies, expected_output_shape",
    [
        (torch.tensor([1]), torch.tensor([4]), torch.Size([1, 1])),
        (torch.tensor([1, 2]), torch.tensor([4, 5]), torch.Size([2, 1])),
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.Size([3, 1])),
    ],
)
def test_factorization_model_forward(model, users, movies, expected_output_shape):
    """Test the forward method of FactorizationModel."""
    output = model(users, movies)
    assert output.shape == torch.Size(expected_output_shape)


def test_factorization_model_embedding_shape(model):
    """Test the dimensions of the user and movie embeddings."""
    assert model.user_embedding.weight.shape == torch.Size(
        [model.num_users, model.embedding_size]
    )
    assert model.movie_embedding.weight.shape == torch.Size(
        [model.num_movies, model.embedding_size]
    )


def test_factorization_model_hidden_layer_shape(model):
    """Test the hidden layers of FactorizationModel."""
    assert model.fc1.weight.shape == torch.Size(
        [model.hidden_dim, 2 * model.embedding_size]
    )
    assert model.fc2.weight.shape == torch.Size([1, model.hidden_dim])


def test_factorization_model_dropout(model):
    """Test the dropout layer of FactorizationModel."""
    assert isinstance(model.dropout, torch.nn.Dropout)
