import torch
from torch import nn

from typing import Callable


def create_default_head(
        input_dim : int,
        output_dim : int,
        activation : Callable,
        dropout: float
    ):
    """Creates the default head module that predicts binned player and ball position.

    Args:
        input_dim (int): Input dimension for the first linear layer.
        output_dim (int): Number of classes to predict.
        activation (Callable): Activation function.
        dropout (float): Dropout rate.

    Returns:
        nn.Module: The classification head.
    """
    class Head(nn.Module):
        def __init__(
            self,
            input_dim : int,
            output_dim : int,
            activation : Callable,
            dropout: float
        ) -> None:
            super().__init__()
            

            self.fc1 = nn.Linear(input_dim, input_dim)
            self.fc2 = nn.Linear(input_dim, output_dim)
            self.activation = activation
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.dropout(x)
            x = self.fc1(x)
            x = self.activation(x)
            return self.fc2(x)

    return Head(
        input_dim,
        output_dim,
        activation,
        dropout
    )