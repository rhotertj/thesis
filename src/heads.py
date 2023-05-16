import torch
from torch import nn
from pytorchvideo.models.head import create_vit_basic_head, SequencePool
from pooling import NetVLAD

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


def create_vit_vlad_head(
    *,
    in_features: int,
    out_features: int,
    n_clusters: int,
    seq_pool_type: str = "cls",
    dropout_rate: float = 0.5,
    activation: Callable = None,
) -> nn.Module:
    """
    Creates vision transformer head with prepended VLAD pooling.
    Aside from VLAD, this code is adapted from pytorchvideo.
    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/head.py
    ::


                                        Pooling
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation


    Activation examples include: ReLU, Softmax, Sigmoid, and None.
    Pool type examples include: cls, mean and none.

    Args:

        in_features: input channel size of the resnet head.
        out_features: output channel size of the resnet head.

        pool_type (str): Pooling type. It supports "cls", "mean " and "none". If set to
            "cls", it assumes the first element in the input is the cls token and
            returns it. If set to "mean", it returns the mean of the entire sequence.

        activation (callable): a callable that constructs vision transformer head
            activation layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and
            None (not applying activation).

        dropout_rate (float): dropout rate.
    """
    assert seq_pool_type in ["cls", "mean", "none"]

    if seq_pool_type in ["cls", "mean"]:
        seq_pool_model = SequencePool(seq_pool_type)
    elif seq_pool_type == "none":
        seq_pool_model = None
    else:
        raise NotImplementedError
    
    if n_clusters > 0:
        vlad_layer = NetVLAD(
            dim=in_features,
            num_clusters=n_clusters
        )
        # in_features = vlad_layer.get_output_size()
    else:
        vlad_layer = None

    if activation is None:
        activation_model = None
    elif activation == nn.Softmax:
        activation_model = activation(dim=1)
    else:
        activation_model = activation()

    return VisionTransformerVLADHead(
        sequence_pool=seq_pool_model,
        vlad_pool=vlad_layer,
        dropout=nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None,
        proj=nn.Linear(in_features, out_features),
        activation=activation_model,
    )


def create_mvit_twin_head(dim_in, num_classes, n_clusters, activation, dropout):
    # this should be Pooling (sequence and vlad) -> heads for regression and classification task
    # use basic twin head, cls and vlad pooling before
    pool = SequencePool("cls")
    if n_clusters > 0:
        vlad_pool = NetVLAD(num_clusters=n_clusters, dim=768)
        dim_in = vlad_pool.get_output_size()
        pool = nn.Sequential(pool, vlad_pool)

    head = MViTTwinHead(
        pool=pool,
        dim_in=dim_in,
        num_classes=num_classes,
        activation=activation,
        dropout=dropout
    )
    
    return head

class VisionTransformerVLADHead(nn.Module):
    """
    Vision transformer head with VLAD pooling.

    ::

                                      SequencePool
                                           ↓
                                        VLADPool
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation


    The builder can be found in `create_vit_vlad_head`.
    """

    def __init__(
        self,
        sequence_pool: nn.Module = None,
        vlad_pool: nn.Module = None,
        dropout: nn.Module = None,
        proj: nn.Module = None,
        activation: nn.Module = None,
    ) -> None:
        """
        Args:
            sequence_pool (torch.nn.modules): pooling module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        self.sequence_pool = sequence_pool
        self.vlad_pool = vlad_pool
        self.dropout = dropout
        self.proj = proj
        self.activation = activation
        assert self.proj is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Performs pooling.
        if self.sequence_pool is not None:
            x = self.sequence_pool(x)

        if self.vlad_pool is not None:
            x_dim = x.shape[-1]
            print(x_dim)
            x = torch.unflatten(self.vlad_pool(x), -1, (x_dim, -1)).mean(-1)

        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        x = self.proj(x)
        # Performs activation.
        if self.activation is not None:
            x = self.activation(x)
        return x

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