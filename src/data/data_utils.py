import numpy as np
from collections import Counter


def combine_teams_with_indicator(team_a: np.ndarray, team_b: np.ndarray) -> np.ndarray:
    """Adds a third dimension to both teams that indicates the team membership of players.
    Returns a stacked array containing both teams.

    Args:
        team_a (np.ndarray): Positions of the first team.
        team_b (np.ndarray): Positions of the second team.

    Returns:
        np.ndarray: Combined teams with team indicator dimension.
    """
    if team_a.shape[2] == 2:
        # add team indicator
        team_a_indicator = np.zeros((*team_a.shape[:2], 1))
        team_b_indicator = team_a_indicator + 1

        team_a = np.concatenate([team_a, team_a_indicator], axis=-1)
        team_b = np.concatenate([team_b, team_b_indicator], axis=-1)
    else:
        # switch dummy z position with team indicator
        team_a[:, :, 2] = 0
        team_b[:, :, 2] = 1

    teams_pos = np.hstack([team_a, team_b])
    return teams_pos


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
        positions[:, :, 1] = court_height - positions[:, :, 1]
    if horizontal:
        positions[:, :, 0] = court_width - positions[:, :, 0]

    return positions


def check_label_within_slice(window_idx, index, sampling_rate):
    """Checks whether an annotation exists for the current window slice that gets overlooked
    because the sampling rate is bigger than 1.

    Args:
        window_idx (int): The current index.
        events (pd.Index): Frame numbers that portray an action.
        sampling_range (int): The sampling rate.

    Returns:
        idx (int): The frame number of the event in the current window slice.
    """
    if sampling_rate == 1:
        return False
    # Resample the entire range with sampling rate 1
    for idx in range(window_idx, window_idx + sampling_rate):
        if idx in index:
            return idx


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