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

    def __init__(self, dim_in, dim_h, num_classes, heads=8):
        super().__init__()
        # add feature embedding layer first?
        self.input_layer = torch.nn.Linear(dim_in, dim_h)
        self.gat1 = GATv2Conv(dim_h, dim_h, num_heads=heads, feat_drop=0.4)
        self.gat2 = GATv2Conv(dim_h, dim_h, num_heads=heads)
        self.relu = torch.nn.ReLU()
        # TODO Configure readout and other params
        # TODO Make mean a separate thing
        self.head = create_default_head(
            input_dim=dim_h,
            output_dim=num_classes,
            activation=self.relu,
            dropout=0.3)

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
        h = dgl.mean_nodes(g, "h")
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
    # https://github.com/facebookresearch/mvit/blob/8ee201520936fad17ca474dadf4114d49945d732/mvit/models/mvit_model.py#L158

    example = ds[1350]
    positions = torch.tensor(example["positions"], dtype=torch.float32)
    epsilon = 7
    
    g = create_graph(positions, epsilon)

    model = GAT(49, 64, 64)
    print(model(g, g.ndata["positions"]))