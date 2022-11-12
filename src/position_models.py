# import pytorch_lightning as pl

import math
import torch
import numpy as np
from torch import nn

# Adapted from https://github.com/airalcorn2/baller2vec/blob/master/baller2vec.py

class Baller2Vec(nn.Module):
    """
    baller2vec: A Multi-Entity Transformer For Multi-Agent Spatiotemporal Modeling
    Michael A. Alcorn, Anh Nguyen
    https://arxiv.org/abs/2102.03291

    Simultaneously models the trajectories of all players
    on the court and the trajectory of the ball.

    """
    def __init__(
        self,
        input_mlp_layers : list[int],
        seq_len : int,
        n_players : int,
        n_player_labels : int,
        n_ball_labels : int,
        nhead : int,
        dim_feedforward : int,
        num_layers : int,
        dropout : float,
    ):
        """Initialize input embedding, transformer and classifier.

        Args:
            input_mlp_layers (list[int]): Linear layer dimensions for transformer input.
            seq_len (int): Number of frames
            n_players (int): Number of players (both teams).
            n_player_labels (int): Number of classes for player position.
            n_ball_labels (int): Number of classes for ball position.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Transformer feed-forward dimension.
            num_layers (int): Number of Transformer blocks.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.seq_len = seq_len
        self.n_players = n_players
        initrange = 0.1

        # Initialize preprocessing MLPs.
        # Extra dimensions for (x, y) coordinates and hoop side (for players) or z
        # coordinate (for ball).
        # (T x 11 + 11 + 1 x 2) => T * 23 x d_model

        in_feats = 3 # x, y, z or x, y, side
        input_mlp = nn.Sequential()
        
        for (layer_idx, out_feats) in enumerate(input_mlp_layers):
            input_mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))
            in_feats = out_feats

        input_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

        self.input_mlp = input_mlp

        # Initialize Transformer.
        d_model = input_mlp_layers[-1]
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Initialize classification layers.
        # HEAD
        self.player_classifier = nn.Linear(d_model, n_player_labels)
        self.player_classifier.weight.data.uniform_(-initrange, initrange)
        self.player_classifier.bias.data.zero_()

        self.ball_classifier = nn.Linear(d_model, n_ball_labels)
        self.ball_classifier.weight.data.uniform_(-initrange, initrange)
        self.ball_classifier.bias.data.zero_()

        # Initialize mask.
        self.register_buffer("mask", self.generate_self_attn_mask())

    def generate_self_attn_mask(self) -> np.ndarray:
        """Generate the attention mask for the transformer to prevent peeking into the future.

        Expects SequenceLength * (NPlayers + Ball) x Hidden Dim
        Assumes [T * Player Feats, T * Ball Feats] 

        Returns:
            np.ndarray: The mask.
        """

        sz = (self.n_players + 1) * self.seq_len

        mask = torch.zeros(sz, sz)
        ball_start = self.n_players * self.seq_len

        for step in range(self.seq_len):
            start = self.n_players * step
            stop = start + self.n_players
            ball_stop = ball_start + step + 1

            # The current players can look at the previous players.
            mask[start:stop, :stop] = 1
            # The current players can look at the current ball.
            mask[start:stop, ball_start:ball_stop] = 1
            # The current ball can look at the previous players.
            mask[ball_start + step, :stop] = 1
            # The current ball can look at the previous balls.
            mask[ball_start + step, ball_start:ball_stop] = 1

        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x : torch.Tensor) -> tuple[torch.tensor, torch.tensor]:
        """Expects SequenceLength * (NPlayers + Ball) x Hidden Dim
        Assumes [T * Player Feats, T * Ball Feats]  

        Args:
            x (torch.Tensor): Positions.

        Returns:
            tuple[torch.tensor, torch.tensor]: Player and Ball predictions
        """
        # X being B x T x 23 x 3

        # Inflate features to transformer dimension
        input_features = self.input_mlp(x) * math.sqrt(self.d_model)
        # Compute representation with masked future frames
        output = self.transformer(input_features, self.mask)

        # Classify next position based on transformer representation        
        players = self.player_classifier(output).squeeze(1)
        ball = self.ball_classifier(output).squeeze(1)

        return players, ball

if "__main__" == __name__:
    x = torch.rand((8 *23, 3)).to("cuda:0")
    model = Baller2Vec(
        [128,512],
        8,
        22,
        1000,
        1000,
        8,
        2048,
        8,
        0.5
    ).to(device="cuda:0")
    out = model(x)
