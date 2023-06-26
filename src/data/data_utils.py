import numpy as np
import torch
from collections import Counter
import dgl
import networkx as nx
import itertools

# TODO: Think about different "styles" of graph 
#   -> connecting player by team and not by proximity
#   -> handcrafted features
class PositionContainer:

    def __init__(
        self,
        team_a: np.ndarray,
        team_b: np.ndarray,
        ball: np.ndarray,
        mirror_vertical: bool = False,
        mirror_horizontal: bool = False,
    ) -> None:
        """Takes positions in [T, N, C] format and preprocesses them. 

        Args:
            team_a (np.ndarray): Positions for team A.
            team_b (np.ndarray): Positions for team B.
            ball (np.ndarray): Ball positions.
            mirror_vertical (bool, optional): Whether to mirror positions vertically. Defaults to False.
            mirror_horizontal (bool, optional): Whether to mirror positions horizontally. Defaults to False.
        """        
        self.team_a = team_a
        self.team_b = team_b
        self.ball = ball

        self.mirror_vertical = mirror_vertical
        self.mirror_horizontal = mirror_horizontal
        self._preprocess_data()
        self.T = self.team_a.shape[-3]
        self.N = 15
        self.max_dist = 44.8 # pythagoras, diagonal of court size 40x20
        self.player_order = torch.arange(0, self.N) # this determines player order and can be altered to shuffle

    def __repr__(self) -> str:
        return f"Position Container for {self.T} time steps ({self.mirror_horizontal=}, {self.mirror_vertical=})"

    def _preprocess_data(self):
        """Truncate or pad teams to size 7, mirror positions if needed. 
        """        
        # ensure correct teamsizes
        self.team_a, self.team_b = ensure_correct_team_size(self.team_a, self.team_b)

        self.team_a = torch.Tensor(self.team_a)
        self.team_b = torch.Tensor(self.team_b)
        self.ball = torch.Tensor(self.ball)
        # Mirror teams and ball if necessary
        self.team_a = mirror_positions_tn3(
            self.team_a,
            vertical=self.mirror_vertical,
            horizontal=self.mirror_horizontal,
        )
        self.team_b = mirror_positions_tn3(
            self.team_b,
            vertical=self.mirror_vertical,
            horizontal=self.mirror_horizontal,
        )
    
        self.ball = mirror_positions_tn3(
            self.ball,
            vertical=self.mirror_vertical,
            horizontal=self.mirror_horizontal,
        )


    def mirror_again(self, horizontal : bool, vertical : bool):
        self.team_a = mirror_positions_tn3(
            self.team_a,
            vertical=vertical,
            horizontal=horizontal,
        )
        self.team_b = mirror_positions_tn3(
            self.team_b,
            vertical=vertical,
            horizontal=horizontal,
        )
    
        self.ball = mirror_positions_tn3(
            self.ball,
            vertical=vertical,
            horizontal=horizontal,
        )
        return self

    def as_graph_per_sequence(self, epsilon: int, relative_positions: bool) -> dgl.DGLGraph:
        """Constructs a graph with flattened trajectory per player as node features.
        Nodes are connected within epsilon neighborhood or team membership if epsilon = 0.

        Args:
            epsilon (int): Neighborhood distance.

        Returns:
            dgl.DGLGraph: The graph with ndata "positions". 
        """        
        positions = self.as_flattened(normalize=False)
        G = dgl.graph([]).to(positions.device)
        G = dgl.add_nodes(G, 15)

        if epsilon > 0:
            # create edges wrt eps nbhood at timestep 0
            for a1, a2 in itertools.combinations(range(15), r=2):
                dist = torch.linalg.norm(positions[a1, 1:2] - positions[a2, 1:2])
                if dist < epsilon:
                    G = dgl.add_edges(G, [a1, a2], [a2, a1])
        else:
            # create edges wrt team membership
            for a1, a2 in itertools.combinations(range(15), r=2):
                # ball to agent
                if a1 == 15 or a2 == 15:
                    G = dgl.add_edges(G, [a1, a2], [a2, a1])

                # same team
                if positions[a1][0] == positions[a2][0]:
                    G = dgl.add_edges(G, [a1, a2], [a2, a1])

        positions[:, 1::3] /= 40  # court length
        positions[:, 2::3] /= 20  # court length
        if relative_positions:
            # subtract ball position from player positions except team indicator
            positions[:-1, 1:] -= positions[-1, 1:]

        G.ndata["positions"] = positions

        G = dgl.add_self_loop(G)
        return G

    def as_graph_per_timestep(self, epsilon: int) -> dgl.DGLGraph:
        """Constructs a graph per timestep with absolute postion as node features.
        Nodes are connected within epsilon neighborhood or team membership if epsilon = 0.

        Args:
            epsilon (int): Neighborhood distance.

        Returns:
            list: List with graphs per timestep. 
        """        
        positions = self.as_TNC(normalize=False)
        graphs = []
        for t in range(positions.shape[0]):
            G = dgl.graph([]).to(positions.device)
            G = dgl.add_nodes(G, 15)
            
            if epsilon > 0:
                # create edges wrt eps nbhood
                for a1, a2 in itertools.combinations(range(15), r=2):
                    # team indicator is a position 0
                    dist = torch.linalg.norm(positions[t, a1, 1:3] - positions[t, a2, 1:3])
                    if dist < epsilon:
                        # TODO: More intelligent to divide by eps?
                        rel_dist = dist / self.max_dist
                        G = dgl.add_edges(G, [a1, a2], [a2, a1], data={"w" : torch.Tensor([rel_dist, rel_dist])})
            else:
                # create edges wrt team membership
                for a1, a2 in itertools.combinations(range(15), r=2):
                    # ball to agent
                    if a1 == 15 or a2 == 15:
                        G = dgl.add_edges(G, [a1, a2], [a2, a1])

                    # same team
                    if positions[t, a1, 0] == positions[t, a2, 0]:
                        G = dgl.add_edges(G, [a1, a2], [a2, a1])

            positions[t, :, 1] /= 40  # court length
            positions[t, :, 2] /= 20  # court length
            G.ndata["positions"] = positions[t]

            G = dgl.add_self_loop(G)
            graphs.append(G)

        return graphs

    def as_TNC(self, normalize=True, add_indicator=True) -> torch.Tensor:
        """Returns the positions with separate dimensions for Time (T), Players (N), 
        and Channel (C).

        The channel dimension is organized as follows: [team, x, y, z].

        Args:
            normalize (bool, optional): Whether to normalize positions relative to court size. Defaults to True.
            add_indicator (bool, optional): Whether to add team indicator to channel. Defaults to True.

        Returns:
            torch.Tensor: Positions in [T, N, C] format.
        """
        all_pos = torch.concatenate([self.team_a, self.team_b, self.ball], dim=-2)
        if normalize:
            all_pos[:, :, 0] /= 40
            all_pos[:, :, 1] /= 20
        if add_indicator:
            # concat agents along agent dimension, add team indicator per agent
            indicator = torch.concatenate([
                torch.zeros(7),
                torch.ones(7),
                torch.Tensor([2])
            ])
            indicator = indicator.repeat(self.T).reshape(self.T, self.N, 1)
            all_pos = torch.concatenate([indicator, all_pos], dim=-1)

        return all_pos

    def as_flattened(self, normalize=True, relative_positions: bool = False, team_indicator: bool = True) -> torch.Tensor:
        """Returns the positions with separate dimension for Players (N) and positions over time (T*C).
        A team indicator per player is inserted at position 0.

        Args:
            normalize (bool, optional):  Whether to normalize positions relative to court size. Defaults to True.

        Returns:
            torch.Tensor: Positions in [N, 1 + (T * C)] format.
        """        
        all_pos = torch.concatenate([self.team_a, self.team_b, self.ball], dim=-2)
        all_pos = torch.einsum("tnc->ntc", all_pos).reshape(15, -1)
        if normalize:
            all_pos[:, 0::3] /= 40
            all_pos[:, 1::3] /= 20
        if relative_positions:
            all_pos[:-1] -= all_pos[-1]
        if team_indicator:
            indicator = torch.concatenate([
                torch.zeros(7),
                torch.ones(7),
                torch.Tensor([2])
            ]).reshape(-1, 1)
            # concat agents along agent dimension, add team indicator per agent
            all_pos = torch.concatenate([indicator, all_pos], dim=-1)

        return all_pos[self.player_order]


def combine_teams_with_indicator(team_a: np.ndarray, team_b: np.ndarray) -> np.ndarray:
    """Adds a third dimension to both teams, flattens the teams along the temporal dimension and adds a bit that indicates the team membership of players.
    Returns a stacked array containing both teams.

    Args:
        team_a (np.ndarray): Positions of the first team.
        team_b (np.ndarray): Positions of the second team.

    Returns:
        np.ndarray: Combined flattened teams with team indicator.
    """
    if team_a.shape[2] == 2:
        # add team indicator
        dummy_z = np.ones((*team_a.shape[:2], 1))
        team_a = np.concatenate([team_a, dummy_z], axis=-1)
        team_b = np.concatenate([team_b, dummy_z], axis=-1)
    teams_pos = np.hstack([team_a, team_b])
    # time player location
    teams_pos = np.einsum("tpl->ptl", teams_pos).reshape(14, -1)
    team_a_indicator = np.zeros((7, 1))
    team_b_indicator = team_a_indicator + 1

    indicator = np.vstack([team_a_indicator, team_b_indicator])
    teams_pos = np.concatenate([indicator, teams_pos], axis=1)
    return teams_pos


def combine_ball_with_indicator(ball_pos: np.ndarray) -> np.ndarray:
    """Adds a third dimension to the ball, flattens it along the temporal dimension and adds a bit that indicates
    that positions belong to the ball.

    Args:
        ball_pos (np.ndarray): Positions of the ball.

    Returns:
        np.ndarray: Flattened ball with indicator.
    """
    # add z dim for ball if not given
    if ball_pos.shape[2] == 2:
        ball_z = np.zeros((ball_pos.shape[0], 1, 1))
        ball_pos = np.concatenate([ball_pos, ball_z], axis=-1)

    ball_pos = ball_pos.reshape(1, -1)
    indicator = np.array([[2]])
    ball_pos = np.concatenate([indicator, ball_pos], axis=1)
    return ball_pos


def ensure_correct_team_size(team_a, team_b):
    """Some trajectory data comes with different team sizes, inflated with zeros and more than 7 active players.
    We downsize all teams to 7 players and pad teams with missing data.

    Args:
        team_a (np.ndarray): Trajectory for the first team.
        team_b (np.ndarray): Trajectory for the second team.

    Returns:
        team_a (np.ndarray): Cleaned trajectory for the first team.
        team_b (np.ndarray): Cleaned trajectory for the second team.
    """

    # TODO: Investigate missing agents, can we interpolate them or
    # pad in "the right" position

    # Iterate timesteps with non-overlapping windows
    # Count appearances of all agents
    # Take most common 7 agents per window
    window_size = max(1, team_a.shape[0] // 4)
    team_agents = []
    for team in (team_a, team_b):
        team_available = np.where(team)
        active_agents = [np.unique(team_available[1][team_available[0] == t]) for t in range(team.shape[0])]
        agents_per_timestep = []  # this will hold the downsized team at every timestep
        cnt = Counter()
        # count unique agents in each window
        for t in range(0, len(active_agents), window_size):
            cnt.clear()
            window_timesteps = 0  # last window is probably smaller than window size
            for t_agents in active_agents[t:t + window_size]:
                cnt.update(t_agents)
                window_timesteps += 1

            common_seven = [a for (a, _) in cnt.most_common(7)]
            agents_per_timestep.extend([common_seven] * window_timesteps)

        team_agents.append(agents_per_timestep)

    agents_a, agents_b = team_agents
    team_a_clean = np.zeros((team_a.shape[0], 7, 3))
    team_b_clean = np.zeros((team_b.shape[0], 7, 3))
    for t in range(team_a.shape[0]):
        # assigning positions to an "empty" array also achieves padding
        team_a_clean[t, 0:len(agents_a[t])] = team_a[t, agents_a[t]]
        team_b_clean[t, 0:len(agents_b[t])] = team_b[t, agents_b[t]]

    return team_a_clean, team_b_clean


def mirror_positions_tn3(
    positions: torch.Tensor,
    horizontal: bool = True,
    vertical: bool = False,
    court_width: int = 40,
    court_height: int = 20
):
    """Mirrors the given positions of players and ball on the court.
    Horizontal mirroring effectively switches sides whereas vertical mirroring
    switches left and right.

    Args:
        positions (torch.Tensor): Player and ball positions.
        horizontal (bool, optional): Mirror horizontally. Defaults to False.
        vertical (bool, optional): Mirror vertically. Defaults to True.
        court_width (int, optional): Court width in meters. Defaults to 40.
        court_height (int, optional): Court height in meters. Defaults to 20.
    """
    new_positions = positions.clone()
    if horizontal:
        new_positions[:, :, 0] = court_width - positions[:, :, 0]
    if vertical:
        new_positions[:, :, 1] = court_height - positions[:, :, 1]

    return new_positions


def mirror_positions(
    positions: np.ndarray,
    horizontal: bool = True,
    vertical: bool = False,
    court_width: int = 40,
    court_height: int = 20
):
    """Mirrors the given positions of players and ball on the court.
    Horizontal mirroring effectively switches sides whereas vertical mirroring
    switches left and right.

    Args:
        positions (np.ndarray): Player and ball positions.
        horizontal (bool, optional): Mirror horizontally. Defaults to False.
        vertical (bool, optional): Mirror vertically. Defaults to True.
        court_width (int, optional): Court width in meters. Defaults to 40.
        court_height (int, optional): Court height in meters. Defaults to 20.
    """
    if vertical:
        positions[:, 2::3] = court_height - positions[:, 2::3]
    if horizontal:
        positions[:, 1::3] = court_width - positions[:, 1::3]

    return positions

def check_label_within_slice(start, stop, events, center_idx, sampling_rate):
    """Checks whether an annotation exists for the current window slice.

    Args:
        start (int): First frame number of the window slice.
        stop (int): Last frame number of the window slice.
        events (pd.DataFrame): The event dataframe.
        center_idx (int): Frame number of the center frame of the window.
        sampling_rate (int): The sampling rate.

    Returns:
        idx (int): The frame number of the event in the current window slice.
    """
    label_idx = -1
    for idx in range(start, stop):
        if idx in events.index:
            label_idx = idx
            break

    if  label_idx == -1:
        label, label_offset = {}, 0
    else:
        label = events.loc[label_idx].labels
        label_offset = (label_idx - center_idx) / sampling_rate
            
    return label, label_offset, label_idx


def get_index_offset(boundaries, idx2frame, idx):
    """The dataset is indexed by frames and positions based on sequence length - not all frames have positional data.
    This function maps the index w.r.t. the dataset to the the match (mapping) and the frame number (offset)

    Args:
        boundaries (List[int]): Dataset indices that belong to the next match
        idx2frame (List[np.array]): Mapping from idx to frame number per match
        idx (int): Index wrt. dataset

    Returns:
        match_number (int): Match number for given index
        frame_idx (int): Frame number for given index
    """
    for match, (i, j) in enumerate(zip(boundaries, boundaries[1:])):
        if i <= idx and idx < j:
            match_number = match
            offset = idx - boundaries[match]
            frame_idx = idx2frame[match][offset]

            return match_number, frame_idx
    raise IndexError(f"{idx} could not be found with boundaries {boundaries}.")


def create_graph(positions: torch.Tensor, epsilon: float) -> dgl.DGLGraph:
    # create graph, fully connected
    G = dgl.graph([]).to(positions.device)
    G = dgl.add_nodes(G, 15)

    # create edges wrt eps nbhood at timestep 0
    for a1, a2 in itertools.combinations(range(15), r=2):
        dist = torch.linalg.norm(positions[a1, 1:2] - positions[a2, 1:2])
        if dist < epsilon:
            G = dgl.add_edges(G, [a1, a2], [a2, a1])

    # normalize positions h w
    positions[:, 1::3] /= 40  # court length
    positions[:, 2::3] /= 20  # court length
    G.ndata["positions"] = positions

    G = dgl.add_self_loop(G)
    return G


if __name__ == "__main__":
    batch_size = 1
    containers = []
    for _ in range(batch_size):
        a, b = torch.rand((16,7,3)), torch.ones((16,7,3)) 
        ball = torch.rand((16,1,3)) + 1
        pos = PositionContainer(a, b, ball)
        print(pos.as_graph_per_sequence(7, True))
        containers.append(pos)

    # batched_container = PositionContainer.from_multiple_containers(containers)
    # print(batched_container.asTNC().shape)
    # G = create_graph(positions, 7)
    # print(G.nodes(data=True))

    # dG = dgl.from_networkx(G, node_attrs=["positions"], idtype=torch.float32)

    # print(dG)