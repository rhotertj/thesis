import torch
from dgl.nn import GATv2Conv
import dgl
import torch.functional as F
import networkx as nx
import itertools
import numpy as np
from data.data_utils import create_graph
from heads import create_default_head


class GAT(torch.nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        num_classes: int,
        input_embedding: bool,
        readout: str = "mean",
        heads: int = 8,
    ):
        """A simple Graph Attention Model.

        Args:
            dim_in (int): Input dimension.
            dim_h (int): Hidden dimension.
            num_classes (int): Number of classes to classify.
            input_embedding (bool): Whether to use a linear layer to project input before GAT layers.
            readout (str, optional): Readout operation. Can be "sum", "min", "mean" and "max". Defaults to "mean".
            heads (int, optional): Number of attention heads. Defaults to 8.
        """    	
        super().__init__()
        if input_embedding:
            self.input_layer = torch.nn.Linear(dim_in, dim_h)
            dim_in = dim_h
        else:
            self.input_layer = torch.nn.Identity()
        self.gat1 = GATv2Conv(dim_in, dim_h, num_heads=heads, feat_drop=0.4)
        self.gat2 = GATv2Conv(dim_h, dim_h, num_heads=heads)
        self.readout = readout
        self.relu = torch.nn.ReLU()

        self.head = create_default_head(input_dim=dim_h, output_dim=num_classes, activation=self.relu, dropout=0.3)

    def forward(self, g, g_feats):
        # graph attention
        g_feats = self.input_layer(g_feats)
        h = self.gat1(g, g_feats)
        # average heads
        h = torch.mean(h, dim=1)
        h = self.relu(h)
        h = self.gat2(g, h)
        h = torch.mean(h, dim=1)
        # readout function (handles batched graph)
        g.ndata["h"] = h
        h = dgl.readout_nodes(g, "h", op=self.readout)
        # classify graph representation
        h = self.head(h)
        h = h.softmax(-1)
        return h


if __name__ == "__main__":
    from data.datasets import MultiModalHblDataset

    ds = MultiModalHblDataset(
        "/nfs/home/rhotertj/datasets/hbl/meta3d_val.csv",
        seq_len=16,
        sampling_rate=2,
        load_frames=False,
    )

    # TODO Add cls token to model and input
    # Might be difficult with batched graph, maybe a custom readout func?
    # might work with topk nodes
    # https://github.com/facebookresearch/mvit/blob/8ee201520936fad17ca474dadf4114d49945d732/mvit/models/mvit_model.py#L158

    example = ds[1350]
    positions = torch.tensor(example["positions"], dtype=torch.float32)
    epsilon = 7

    g = create_graph(positions, epsilon)

    model = GAT(49, 64, 64)
    print(model(g, g.ndata["positions"]))