import torch
from dgl.nn import GATv2Conv
import dgl
import torch.nn.functional as F
import networkx as nx
import itertools
import numpy as np
from data.data_utils import create_graph
from heads import create_default_head

# also try multiple graphs and temporal pooling afterwards
#   -> we would need to alter collate (?)
# position transformer

class GAT(torch.nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        num_classes: int,
        input_operation: str,
        batch_size : int,
        readout: str = "mean",
        num_heads: int = 8,
        use_head: bool = True,
    ):
        """A simple Graph Attention Model.

        Args:
            dim_in (int): Input dimension.
            dim_h (int): Hidden dimension.
            num_classes (int): Number of classes to classify.
            input_embedding (str): Operation to apply before graph layers. Can be one of "linear", "raw", "conv", "conv+linear", "conv->linear".
            readout (str, optional): Readout operation. Can be "sum", "min", "mean" and "max". Defaults to "mean".
            heads (int, optional): Number of attention heads. Defaults to 8.
        """    	
        super().__init__()
        self.use_head = use_head

        # self.input_layer = torch.nn.Linear(dim_in, dim_h)
        # self.gat1 = GATv2Conv(dim_h, dim_h, num_heads=num_heads, feat_drop=0.4)
        self.input_layer = InputLayer(dim_in=dim_in, dim_h=dim_h, op=input_operation, batch_size=batch_size)
        self.gat1 = GATv2Conv(self.input_layer.get_output_size(), dim_h, num_heads=num_heads, feat_drop=0.4)
        self.gat2 = GATv2Conv(dim_h, dim_h, num_heads=num_heads)
        self.readout = readout
        self.relu = torch.nn.ReLU()
        self.head = create_default_head(input_dim=dim_h, output_dim=num_classes, activation=self.relu, dropout=0.3)
        

    def forward(self, g, g_feats):
        # graph attention
        # g_feats = self.input_layer(g_feats)
        h = self.input_layer(g_feats)
        h = self.gat1(g, h)
        # average heads
        h = torch.mean(h, dim=1)
        h = self.relu(h)
        h = self.gat2(g, h)
        h = torch.mean(h, dim=1)
        # readout function (handles batched graph)
        g.ndata["h"] = h
        h = dgl.readout_nodes(g, "h", op=self.readout)
        # classify graph representation
        if self.use_head:
            h = self.head(h)
            h = h.softmax(-1)
        return h


class GCN(torch.nn.Module):
    # graph conv layer module
    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        num_classes: int,
        input_operation: str,
        batch_size : int,
        readout: str = "mean",
        num_heads: int = 8,
        use_head: bool = True,
    ):

        pass

    def forward(self, g, g_feats):
        # graph attention
        # g_feats = self.input_layer(g_feats)
        h = self.input_layer(g_feats)
        h = self.gat1(g, h)
        # average heads
        h = torch.mean(h, dim=1)
        h = self.relu(h)
        h = self.gat2(g, h)
        h = torch.mean(h, dim=1)
        # readout function (handles batched graph)
        g.ndata["h"] = h
        h = dgl.readout_nodes(g, "h", op=self.readout)
        # classify graph representation
        if self.use_head:
            h = self.head(h)
            h = h.softmax(-1)
        return h

class PositionTransformer(torch.nn.Module):
    
    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        num_classes: int,
        batch_size : int,
        num_heads: int = 8,
        use_head: bool = True,
    ):
        # encoder and head
        super().__init__()
        self.use_head = use_head

        self.linear = torch.nn.Linear(dim_in, dim_h)
        layer = torch.nn.TransformerEncoderLayer(d_model=dim_h, nhead=num_heads)
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=6)
        self.head = create_default_head(input_dim=dim_h, output_dim=num_classes, activation=F.relu, dropout=0.3)

    def forward(self, x):
        y = self.linear(x)
        y = self.transformer(y)
        y = y.mean(dim=1)
        if self.use_head:
            y = self.head(y)
        return y


class InputLayer(torch.nn.Module):
    # get type of operation from model config

    def __init__(self, dim_in, dim_h, op, batch_size) -> None:
        super().__init__()
        self.op = op
        # kernel_size=6, stride=3 means we convolve over adjacent (x,y,z) positions
        kernel = 6
        stride = 3
        channels = batch_size * (7+7+1) # both teams and ball
        match op:
            case "linear":
                self.linear = torch.nn.Linear(dim_in, dim_h)
                self.out_size = dim_h
            case "raw":
                self.identity = torch.nn.Identity()
                self.out_size = dim_in
            case "conv":
                self.conv =  torch.nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel, stride=stride)
                self.out_size = ((dim_in - kernel) // stride) + 1 + 1 # conv result + team indicator
            case "conv+linear":
                self.conv =  torch.nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=6, stride=3)
                self.linear = torch.nn.Linear(dim_in - 1, dim_h)
                self.out_size = ((dim_in - kernel) // stride) + 1 + dim_h + 1 # conv result + team indicator + linear output
            case "conv->linear":
                self.conv =  torch.nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=6, stride=3)
                self.linear = torch.nn.Linear(((dim_in - kernel) // stride) + 1 + 1, dim_h) # add team indicator
                self.out_size = dim_h

    def get_output_size(self):
        return self.out_size

    def __repr__(self):
        return f"Input layer with {self.op=} and {self.out_size=}"

    def forward(self, x):
        match self.op:
            case "linear":
                out = self.linear(x)

            case "raw":
                out = self.identity(x)

            case "conv":
                raw_pos = x[:, 1:]
                team_indicator = x[:, 0].unsqueeze(1)
                local_feats = self.conv(raw_pos)
                out = torch.concat([team_indicator, local_feats], dim=-1)

            case "conv+linear":
                raw_pos = x[:, 1:]
                team_indicator = x[:, 0].unsqueeze(1)
                local_feats = self.conv(raw_pos)
                global_feats = self.linear(raw_pos)
                out = torch.concat([team_indicator, local_feats, global_feats], dim=-1)

            case "conv->linear":
                raw_pos = x[:, 1:]
                team_indicator = x[:, 0].unsqueeze(1)
                local_feats = self.conv(raw_pos)
                out = torch.concat([team_indicator, local_feats], dim=-1)
                out = self.linear(out)


        return out


if __name__ == "__main__":
    # TODO Add cls token to model and input
    # Might be difficult with batched graph, maybe a custom readout func?
    # might work with topk nodes
    # https://github.com/facebookresearch/mvit/blob/8ee201520936fad17ca474dadf4114d49945d732/mvit/models/mvit_model.py#L158

    from data.datasets import MultiModalHblDataset, ResampledHblDataset
    from lit_data import collate_function_builder
    from torchvision import transforms as t
    import video_transforms as vt
    import pytorchvideo.transforms as ptvt

    basic_transforms = t.Compose([
            vt.FrameSequenceToTensor(),
            t.Resize((224,224))
            ])

    collate_fn_flat = collate_function_builder(7, True, None, "flattened")
    collate_fn_graph = collate_function_builder(7, True, None, "graph_per_sequence")

    dataset = MultiModalHblDataset("/nfs/home/rhotertj/datasets/hbl/meta3d.csv", 16, sampling_rate=4, transforms=basic_transforms, overlap=False, load_frames=False)

    instances = []
    for i in range(8):
        x = dataset[i*25]
        instances.append(x)

    flat_batch = collate_fn_flat(instances)
    graph_batch = collate_fn_graph(instances)

    model = PositionTransformer(49, 128, 3, 8, True)
    print(flat_batch["positions"].shape)
    print(model(flat_batch["positions"]))

    gmodel = GAT(49, 128, 3, "linear", 8)
    print(graph_batch["positions"].ndata["positions"].shape)
    print(gmodel(graph_batch["positions"], graph_batch["positions"].ndata["positions"]))
    