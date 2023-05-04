import torch
import torch.nn as nn
from dgl.nn import GATv2Conv, GraphConv
import dgl
import torch.nn.functional as F
import networkx as nx
import itertools
import numpy as np
from data.data_utils import create_graph
from heads import create_default_head, BasicTwinHead
from pooling import MeanPool

class GAT(torch.nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        num_classes: int,
        input_operation: str,
        batch_size : int,
        num_layers: 5,
        readout: str = "mean",
        num_heads: int = 8,
        use_head: bool = True,
    ):
        """A simple Graph Attention Model.

        Args:
            dim_in (int): Input dimension.
            dim_h (int): Hidden dimension.
            num_classes (int): Number of classes to classify.
            input_operation (str): Operation to apply before graph layers. Can be one of "linear", "raw", "conv", "conv+linear", "conv->linear".
            batch_size (int): Batch size used for training. This is relevant to correctly size input layer.
            num_layers (int): Number of GAT layers.
            readout (str, optional): Readout operation. Can be "sum", "min", "mean" and "max". Defaults to "mean".
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            use_head (bool): Whether to use the head for classification or just return the graph representation. Defaults to True.
        """    	
        super().__init__()
        self.use_head = use_head
        self.mean_pool = MeanPool(dim=1)

        self.input_layer = InputLayer(dim_in=dim_in, dim_h=dim_h, op=input_operation, batch_size=batch_size)
        self.gat_layers = [GATv2Conv(self.input_layer.get_output_size(), dim_h, num_heads=num_heads, feat_drop=0.4)]

        for _ in range(num_layers):
            gat_layer = GATv2Conv(dim_h, dim_h, num_heads=num_heads, activation=F.leaky_relu)
            self.gat_layers.append(gat_layer)
            
        self.gat_layers = nn.ModuleList(self.gat_layers)
        self.readout = readout
        self.relu = torch.nn.ReLU()
        self.head = create_default_head(input_dim=dim_h, output_dim=num_classes, activation=self.relu, dropout=0.3)
        

    def forward(self, g, g_feats):
        h = self.input_layer(g_feats)
        # graph attention
        for layer in self.gat_layers:
            h = layer(g, h)
            h = self.mean_pool(h)
        # readout function (handles batched graph)
        g.ndata["h"] = h
        h = dgl.readout_nodes(g, "h", op=self.readout)
        # classify graph representation
        if self.use_head:
            h = self.head(h)
            h = h.softmax(-1)
        return h

class GIN_MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

class GIN(nn.Module):

    def __init__(self,
        dim_in,
        dim_h,
        num_classes,
        learn_eps,
        batch_size,
        input_operation,
    ):
        super().__init__()

        self.input_layer = InputLayer(dim_in, dim_h, op=input_operation, batch_size=batch_size)
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        
        for i in range(num_layers - 1):
            if i==0:
                mlp = GIN_MLP(self.input_layer.get_output_size(), dim_h, dim_h)
            else: 
                mlp = GIN_MLP(dim_h, dim_h, dim_h)
            self.ginlayers.append(
                dgl.nn.GINConv(mlp, learn_eps=learn_eps)
            )
            self.batch_norms.append(nn.BatchNorm1d(dim_h))
 
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        self.linear_prediction.append(
            nn.Linear(self.input_layer.get_output_size(), num_classes)
        )
        for _ in range(num_layers):
            self.linear_prediction.append(nn.Linear(dim_h, num_classes))
        self.drop = nn.Dropout(0.5)
        self.pool = dgl.nn.glob.SumPooling()  # AvgPooling on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        h = self.input_layer(h)
        hidden_reprs = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_reprs.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_reprs):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer

class PositionTransformer(torch.nn.Module):
    
    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        num_classes: int,
        input_operation : str,
        batch_size : int,
        num_heads: int = 8,
        head_type: str = "classify",
    ):
        super().__init__()
        # TODO: Class token with cls pooling
        self.input_layer = InputLayer(dim_in, dim_h, op=input_operation, batch_size=batch_size)
        dim_h = self.input_layer.get_output_size() # must be divisible by num_heads
        layer = torch.nn.TransformerEncoderLayer(d_model=dim_h, nhead=num_heads)
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=6)
        if head_type == "classify":
            self.head = create_default_head(input_dim=dim_h, output_dim=num_classes, activation=F.relu, dropout=0.3)
        if head_type == "twin":
            self.head = BasicTwinHead(dim_in=dim_h, num_classes=num_classes, activation=F.relu, dropout=0.3)
        self.head_type = head_type
        

    def forward(self, x):
        y = self.input_layer(x)
        y = self.transformer(y)
        y = y.mean(dim=1)
        if not self.head_type == "pool":
            y = self.head(y)
        return y

class GCN(torch.nn.Module):

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
    ) -> None:
        super().__init__()
        self.use_head = use_head
        self.readout = readout
        # note that batch size actually denotes sequence length
        self.input_layer = InputLayer(dim_in=dim_in, dim_h=dim_h, op=input_operation, batch_size=1)
        self.gcn1 = GraphConv(
            in_feats=self.input_layer.get_output_size(),
            out_feats=dim_h
        )
        self.gcn2 = GraphConv(
            in_feats=dim_h,
            out_feats=dim_h
        )
        layer = torch.nn.TransformerEncoderLayer(d_model=dim_h, nhead=num_heads)
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.head = create_default_head(dim_h, num_classes, F.relu, 0.3)

    def forward(self, g, g_feats, edge_weights=None):
        # NOTE: This function uses a batched graph for one sequence, one graph per timestep.
        h = self.input_layer(g_feats)

        h = self.gcn1(g, h, edge_weight=edge_weights)
        h = self.gcn2(g, h, edge_weight=edge_weights)

        g.ndata["h"] = h
        h = dgl.readout_nodes(g, "h", op=self.readout)

        g_repr = self.transformer(h)
        g_repr = h.mean(dim=0)

        if not self.use_head:
            return g_repr
        return self.head(g_repr).unsqueeze(0) # unsqueeze for cross entropy loss
        


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
    