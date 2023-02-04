import numpy as np
import pickle as pkl
import pandas as pd
from pathlib import Path
import cv2
from typing import Union
import torch
from collections import Counter
from torch.utils.data import Dataset
from torchvision.transforms import Compose

# TODO: Prepare different representation for positions
# -> On the fly plotting might be too slow

# TODO: Speed dataloading up by:
#   - Putting team downsizing in pre-processing
#   - Putting team indicator in pre-processing


class MultiModalHblDataset(Dataset):

    def __init__(
        self,
        meta_path: str,
        seq_len: int = 8,
        sampling_rate: int = 1,
        load_frames: bool = True,
        transforms: Union[None, Compose] = None,
        label_mapping: callable = lambda x: x,
    ):
        """
        This dataset provides a number of video frames (determined by seq_len) and
        corresponding positional data for ball and players.
        If available, it also provides action label and its position in the current frame stack.

        Note that by increasing the sampling rate, the temporal range of the sampled frames increases,
        whereas the sequence length stays fixed.

        Args:
            meta_path (str): Path to the .csv file that holds paths to frames, annotations and positions.
            seq_len (int, optional): An even desired number of frames. Defaults to 8.
            sampling_rate (int, optional): Sample every nth frame. When set to 1, we sample subsequent frames
                corresponding to seq_len. Defaults to 1.
            load_frames (bool, optional): Whether to read and process images. Defaults to True.
            transforms (Union[None, Compose], optional): Transforms to apply to the video frames. Defaults to None.
            label_mapping (callable optional): A function that maps the label dictionary to an integer. 
                Defaults to the identity function.
        """
        super(MultiModalHblDataset).__init__()

        self.seq_len = (seq_len // 2) * 2
        self.seq_half = seq_len // 2
        self.load_frames = load_frames
        self.sampling_rate = sampling_rate
        self.transforms = transforms
        self.label_mapping = label_mapping

        # might be useful to pass information from pytorch lightning config about dataset
        self.meta = {}

        # print(f"Read {meta_path}...")
        self.meta_df = pd.read_csv(meta_path)
        self.index_tracker = [0]
        self.idx_to_frame_number = []
        self.frame_paths = []
        self.event_dfs = []
        self.position_arrays = []
        self.mirror_vertical = []
        self.mirror_horizontal = []
        for _, row in self.meta_df.iterrows():
            self.frame_paths.append(row["frames_path"])

            event_df = pd.read_csv(row["events_path"], index_col="t_start")
            event_df["labels"] = event_df.labels.apply(eval)  # parse dict from string
            self.event_dfs.append(event_df)

            with open(row["positions_path"], "rb") as f:
                self.position_arrays.append(pkl.load(f))

            self.mirror_horizontal.append(row["mirror_horizontal"])
            self.mirror_vertical.append(row["mirror_vertical"])

        # NOTE: All events, positions and image paths are indexed based on frame number.
        # We need to create a mapping from frame with available positions
        # to absolute frame number to access the data.
        for *_, availability in self.position_arrays:
            sample_range = self.seq_len * self.sampling_rate
            kernel = np.ones(sample_range)
            # Make sure that we have positions along the whole sequence.
            available_windows = np.convolve(availability, kernel)
            available_windows = np.where(available_windows == sample_range)[0]
            # shift index to beginning of sequence
            available_windows -= (sample_range - 1)

            self.idx_to_frame_number.append(available_windows)
            # Only the first half is annotated. We need to create a boundary before the last action.
            # We take the third last since we have annotated actions when position tracking was already paused.
            last_event = self.event_dfs[len(self.index_tracker) - 1].index[-3] - self.seq_len
            last_window = np.where(available_windows == last_event)[0][0]
            self.index_tracker.append(self.index_tracker[-1] + len(available_windows[:last_window]))

    def __len__(self):
        """Length of the dataset over all matches.

            Sliding window approach needs a full sequence as padding,
            half in the beginning and half at the end.

        Returns:
            int: Dataset length.
        """
        return self.index_tracker[-1] - (self.seq_len * self.sampling_rate)

    def __getitem__(
        self,
        idx,
        frame_idx: Union[None, int] = None,
        match_number: Union[None, int] = None,
        positions_offset: int = 0
    ) -> dict:
        """This method returns 
            - stacked frames (RGB) of shape [T x H x W x C]
            - corresponding positional data for players and ball
            - the corresponding event label
            - the annotated event position in the frame stack.

        Note that the provided index is valid over the whole dataset.
        It can not be treated as a frame number or index for a given match.
        It first needs to be matched against a match and a frame number, based on the availability of positional data.

        If we need to access a specific set of frames, we can access them by using the keyword arguments. 

        Args:
            idx (int): Index of data to be retrieved.
            frame_idx (Union[None, int]): Raw frame number relative to match begin. Defaults to None.
            match_number (Union[None, int]): Match number according to meta file. Defaults to None.
            positions_offset (int): Offset between position frame and video frame. Defaults to 0.

        Returns:
            dict: A dict containing frames, positions, label and label_offset.
        """
        # for in loop does not care for size of iterator but goes on until index error is raised
        if idx >= len(self):
            raise IndexError(f"{idx} out of range for dataset size {len(self)}")

        if frame_idx is None:
            # Get correct match based on idx (match_number) and idx with respect to match and availability (frame_idx)
            # from idx with respect to dataset (param: idx)
            # Add half sequence length to index to avoid underflowing dataset idx < seq_len
            match_number, frame_idx = get_index_offset(self.index_tracker, self.idx_to_frame_number, idx)
            frame_idx += (self.seq_half * self.sampling_rate)

        frame_base_path = Path(self.frame_paths[match_number])
        events = self.event_dfs[match_number]
        frames = []
        window_indices = []  # debug information

        label = {}  # Default to 'background' action
        label_offset = 0  # Which frame of the window is portraying the action

        # Iterate over window, load frames and check for event
        half_range = self.seq_half * self.sampling_rate
        for window_idx in range(frame_idx - half_range, frame_idx + half_range, self.sampling_rate):
            window_indices.append(window_idx)
            if window_idx in events.index:
                label = events.loc[window_idx].labels
                label_offset = ((window_idx - frame_idx) + half_range) // self.sampling_rate
            # Check whether we missed an annotation because of a higher sampling rate
            else:
                label_idx = check_label_within_slice(window_idx, events.index, self.sampling_rate)
                if label_idx:
                    label = events.loc[label_idx].labels
                    label_offset = ((window_idx - frame_idx) + half_range) // self.sampling_rate

            if self.load_frames:
                frame_path = str(frame_base_path / f"{str(window_idx).rjust(6, '0')}.jpg")
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        if self.load_frames:
            frames = np.stack(frames)
            if self.transforms:
                frames = self.transforms(frames)

        team_a_pos, team_b_pos, ball_pos, _ = self.position_arrays[match_number]

        position_slice = slice(
            (frame_idx - half_range) - positions_offset,
            (frame_idx + half_range) - positions_offset,
            self.sampling_rate,
        )
        team_a_pos = team_a_pos[position_slice]
        team_b_pos = team_b_pos[position_slice]
        ball_pos = ball_pos[position_slice]

        team_a_pos, team_b_pos = ensure_correct_team_size(team_a_pos, team_b_pos)

        if team_a_pos.shape[2] == 2:
            # add team indicator
            team_a_indicator = np.zeros((*team_a_pos.shape[:2], 1))
            team_b_indicator = team_a_indicator + 1

            team_a_pos = np.concatenate([team_a_pos, team_a_indicator], axis=-1)
            team_b_pos = np.concatenate([team_b_pos, team_b_indicator], axis=-1)
        else:
            # switch dummy z position with team indicator
            team_a_pos[:, :, 2] = 0
            team_b_pos[:, :, 2] = 1

        teams_pos = np.hstack([team_a_pos, team_b_pos])

        # add z dim for ball if not given
        if ball_pos.shape[2] == 2:
            ball_z = np.zeros((ball_pos.shape[0], 1, 1))
            ball_pos = np.concatenate([ball_pos, ball_z], axis=-1)

        all_pos = np.hstack([teams_pos, ball_pos])

        all_pos = mirror_positions(
            all_pos,
            vertical=self.mirror_vertical[match_number],
            horizontal=self.mirror_horizontal[match_number],
        )

        label = self.label_mapping(label)

        instance = {
            "frames": frames,
            "positions": all_pos,
            "label": label,
            "label_offset": label_offset,
            "frame_idx": frame_idx,
            "query_idx": idx,
            "window_indices": window_indices,
            "match_number": match_number
        }
        return instance

    def export_json(self, idx):
        """Generate trajectory in json format for internal visualization tool.
        """
        positions = self[idx]["positions"]
        team_a = positions[:, :7].tolist()
        team_b = positions[:, 7:14].tolist()
        ball = positions[:, 14].tolist()

        jd = {}
        jd["team_a"] = team_a
        jd["team_b"] = team_b
        jd["balls"] = ball
        return jd


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


class LabelDecoder:

    def __init__(self, num_classes: int) -> None:
        """Infers class names and their integer mapping
        from the number of classes. 

        Args:
            num_classes (int): Number of classes.
        """
        self.num_classes = num_classes
        self.class_names = self.get_classnames()
        self.decode_event = self.choose_label_mapping()

    def __call__(self, x: dict) -> int:
        """Maps the input to an integer.

        Args:
            x (dict): The event annotation.

        Returns:
            int: Integer label.
        """
        return self.decode_event(x)

    def get_classnames(self) -> list[str]:
        """Returns the classnames based on the number of classes.

        Returns:
            list[str]: Class names.
        """
        if self.num_classes == 2:
            return ["Background", "Action"]

        cls_names = ["Background", "Pass", "Shot", "Foul"]
        return cls_names[:self.num_classes]

    def choose_label_mapping(self) -> callable:
        """Choose correct function to decode event annotations based on the number of classes.

        Raises:
            ValueError: Unknown number of classes.

        Returns:
            callable: Function that maps annotations to integers.
        """
        if self.num_classes == 2:
            return self.has_action
        if self.num_classes == 3:
            return self.background_pass_shot
        if self.num_classes == 4:
            return self.background_pass_shot_foul
        else:
            raise ValueError(f"Number of classes ({self.num_classes}) is invalid!")

    def has_action(self, x: dict):
        """Decodes actionness.

        Args:
            x (dict): The event annotation.

        Returns:
            int: Integer label.
        """
        if x == {}:
            return 0
        return 1

    def background_pass_shot_foul(self, x):
        """Decodes pass, shots and foul.

        Args:
            x (dict): The event annotation.

        Returns:
            int: Integer label.
        """
        if x["Pass"] == "O" and x["Wurf"] == "0":
            return 3
        return self.background_pass_shot(x)

    def background_pass_shot(self, x):
        """Decodes passes and shots.

        Args:
            x (dict): The event annotation.

        Returns:
            int: Integer label.
        """
        if x == {}:
            return 0
        if not x["Pass"] in ("O", "X"):
            pass_labels = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}
            return pass_labels[x["Pass"]]
        elif not x["Wurf"] == "0":
            shot_labels = {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 2}
            return shot_labels[x["Wurf"]]
        return 0


if "__main__" == __name__:
    data = MultiModalHblDataset(
        "/nfs/home/rhotertj/datasets/hbl/meta3d.csv", seq_len=16, sampling_rate=4, load_frames=True
    )
    from utils import array2gif, draw_trajectory
    idx = 7560
    # 18413 is bg should be shot,
    # 18388 is shot, same a 18413#
    # 8998 is preparing for running towards center nothing special (no annot)
    # 13797 paused game after foul (no annotation)
    # 7560 annotated pass but not recognizable

    instance = data[idx]
    # instance = data.__getitem__(idx, frame_idx=16029, match_number=4)
    # data.export_json(idx)
    # exit(0)
    print("All good")
    # breaks at 745124
    # from tqdm import tqdm
    # for i in tqdm(range(len(data))):
    #     try:
    #         data[i]
    #     except:
    #         print(i)
    #         raise IndexError
    frames = instance["frames"]
    positions = instance["positions"]
    label = instance["label"]
    label_offset = instance["label_offset"]
    window = instance['window_indices']

    print(label, label_offset)
    print(f"Frame {instance['frame_idx']} in window {window}")
    print(f"Annotated action is {label} at frame {window[label_offset]}")
    array2gif(frames, f"./img/instance_{instance['query_idx']}.gif", 2)

    fig = draw_trajectory(positions)
    fig.savefig(f"./img/instance_{instance['query_idx']}.png")
