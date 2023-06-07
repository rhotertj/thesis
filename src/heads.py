import torch
from torch import nn
from pytorchvideo.models.head import create_vit_basic_head, SequencePool

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
            return self.fc2(x).softmax(-1)

    return Head(
        input_dim,
        output_dim,
        activation,
        dropout
    )

def create_linear_head(
        input_dim : int,
        output_dim : int,
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
            dropout: float
        ) -> None:
            super().__init__()
            
            self.fc1 = nn.Linear(input_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.dropout(x)
            return self.fc1(x).softmax(-1)

    return Head(
        input_dim,
        output_dim,
        activation,
        dropout
    )

def create_mvit_twin_head(dim_in, num_classes, activation, dropout):
    pool = SequencePool("cls")

    head = MViTTwinHead(
        pool=pool,
        dim_in=dim_in,
        num_classes=num_classes,
        activation=activation,
        dropout=dropout
    )
    
    return head

class BasicTwinHead(nn.Module):

    def __init__(self, dim_in : int, num_classes : int, activation : callable, dropout : float) -> None:
        super().__init__()
        self.cls_head = create_default_head(dim_in, num_classes, activation, dropout)
        self.reg_head = create_default_head(dim_in, 1, activation, dropout)

    def forward(self, x):
        cls = self.cls_head(x)
        offset = self.reg_head(x)
        return cls, offset


class MViTTwinHead(nn.Module):

    def __init__(self, pool : nn.Module, dim_in : int, num_classes : int, activation : callable, dropout : float) -> None:
        super().__init__()
        # dim in is pool sensitive
        self.pool = pool

        self.cls_head = create_vit_basic_head(
            in_features=dim_in,
            out_features=num_classes,
            seq_pool_type="none",
            dropout_rate=0.5,
            activation=torch.nn.Softmax,
        )
        self.reg_head = create_default_head(
            input_dim=dim_in,
            output_dim=1,
            activation=activation,
            dropout=dropout
        )

    def forward(self, x):
        x = self.pool(x)
        cls = self.cls_head(x)
        offset = self.reg_head(x)
        return cls, offset