import numpy as np
import pickle as pkl
import pandas as pd
from pathlib import Path
import cv2
from typing import Union
from torch.utils.data import Dataset

# TODO: Prepare different representation for positions
        # -> On the fly plotting might be too slow

class HandballSyncedDataset(Dataset):

    def __init__(self, meta_path : str, seq_len : int = 8, sampling_rate : int = 1, load_frames : bool = True):
        """
        This dataset provides a number of video frames (determined by seq_len) and
        corresponding positional data for ball and players.
        If available, it also provides action label and its position in the current frame stack.

        Note that by increasing the sampling rate, the temporal range of the sampled frames increases,
        whereas the sequence length stays fixed.

        Args:
            meta_path (str): Path to the .csv file that holds paths to frames, annotations and positions.
            seq_len (int, optional): An even desired number of frames. Defaults to 8.
            sampling_rate (int, optional): Sample every nth frame. When set to 1, we sample subsequent frames corresponding to seq_len. Defaults to 1.
            load_frames (bool, optional): Whether to read and process images. Defaults to True.
        """
        super(HandballSyncedDataset).__init__()

        self.seq_len = seq_len
        self.seq_half = seq_len // 2
        self.load_frames = load_frames
        self.sampling_rate = sampling_rate

        # might be useful to pass information from pytorch lightning config about dataset
        self.meta = {}

        print(f"Read {meta_path}...")
        self.meta_df = pd.read_csv(meta_path)
        self.index_tracker = [0]
        self.idx_to_frame_number = []
        self.frame_paths = []
        self.event_dfs = []
        self.position_arrays = []
        self.position_offsets = []
        for _, row in self.meta_df.iterrows():
            self.frame_paths.append(row["frames_path"])

            event_df = pd.read_csv(row["events_path"], index_col="t_start")
            event_df["labels"] = event_df.labels.apply(eval) # parse dict from string
            self.event_dfs.append(event_df)

            with open(row["positions_path"], "rb") as f:
                self.position_arrays.append(
                    pkl.load(f)
                )
            
            # self.position_offsets.append(int(row["position_offset"] * self.sampling_rate))
        
        # NOTE: All events, positions and image paths are indexed based on frame number.
        # We need to create a mapping from frame with available positions
        # to absolute frame number to access the data.
        for *_, availability in self.position_arrays:
            sample_range = self.seq_len * self.sampling_rate
            kernel = np.ones(sample_range)
            available_windows_cvn = np.convolve(availability, kernel) # previously used mode='valid'
            available_windows = np.where(available_windows_cvn == sample_range)[0] - (sample_range - 1)

            self.idx_to_frame_number.append(available_windows)
            self.index_tracker.append(int(self.index_tracker[-1] + len(available_windows)))


    def __len__(self):
        """Length of the dataset over all matches.
            
            Sliding window approach needs a full sequence as padding,
            half in the beginning and half at the end.

        Returns:
            int: Dataset length.
        """
        return self.index_tracker[-1] - (self.seq_len * self.sampling_rate)

    def __getitem__(self, idx, frame_idx : Union[None, int] = None, match_number : Union[None, int] = None, positions_offset : int = 0):
        """This method returns 
            - stacked frames (RGB) of shape [F x H x W x C]
            - corresponding positional data for players and ball
            - the corresponding event label
            - the annotated event position in the frame stack.

        Note that the provided index is valid over the whole dataset. It can not be treated as a frame number or index for a given match.
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
        if idx >= len(self): raise IndexError(f"{idx} out of range for dataset size {len(self)}")

        
        if frame_idx is None:
            # Get correct match based on idx (match_number) and idx with respect to match and availability (frame_idx) 
            # from idx with respect to dataset (param: idx)
            # Add half sequence length to index to avoid underflowing dataset idx < seq_len
            match_number, frame_idx = get_index_offset(self.index_tracker, self.idx_to_frame_number, idx + (self.seq_half * self.sampling_rate))
            
        # if positions_offset == 0:
        #     positions_offset = self.position_offsets[match_number]
            # print(f"{idx=} {match_number=} {frame_idx=} {positions_offset=}")

        frame_base_path = Path(self.frame_paths[match_number])
        events = self.event_dfs[match_number]
        frames = []
        
        label = 0 # Default to 'background' action
        label_offset = 0 # Which frame of the window is portraying the action

        # Iterate over window, load frames and check for event
        half_range = self.seq_half * self.sampling_rate
        for window_idx in range(frame_idx - half_range, frame_idx + half_range, self.sampling_rate):
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

        team_a_pos, team_b_pos, ball_pos, _ = self.position_arrays[match_number]
        
        position_slice = slice((frame_idx - half_range) - positions_offset, (frame_idx + half_range) - positions_offset, self.sampling_rate)
        team_a_pos = team_a_pos[position_slice]
        team_b_pos = team_b_pos[position_slice]
        ball_pos = ball_pos[position_slice]
        
        team_a_pos, team_b_pos = ensure_equal_teamsize(team_a_pos, team_b_pos)
        
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
        
        if (ball_pos == 0).all():
            ball_pos = ball_pos[: ,0 ,:]
        else:
            ball_avail = np.where(ball_pos)
            ball_pos = ball_pos[ball_avail]
            ball_pos = ball_pos.reshape(self.seq_len, 1, -1)
        

        # add z dim for ball if not given
        if ball_pos.shape[2] == 2:
            ball_z = np.zeros((ball_pos.shape[0], 1, 1))
            ball_pos = np.concatenate([ball_pos, ball_z], axis=-1)
        
        all_pos = np.hstack([teams_pos, ball_pos])

        # all_pos = mirror_positions(all_pos)

        instance = {
            "frames" : frames,
            "positions" : all_pos,
            "label" : label, 
            "label_offset" : label_offset
        }
        return instance

    def export_json(self, idx):
        """Generate trajectory in json format for internal visualization tool.
        """
        raise NotImplementedError()


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
    for idx in range(window_idx - sampling_rate, window_idx + sampling_rate):
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
    

def ensure_equal_teamsize(team_a, team_b):
    """Some positional data come with different team sizes.
    We pad the smaller team with zeros to be able to concatenate the team positions later.

    Args:
        team_a (np.array): Positional data for the first team.
        team_b (np.array): Positional data for the second team.

    Returns:
        team_a (np.array): Padded positional data for the first team.
        team_b (np.array): Padded positional data for the second team.
    """
    if team_a.shape == team_b.shape:
        return team_a, team_b
    
    pad_size = team_a.shape[1] - team_b.shape[1]

    if pad_size < 0: # team b has more players
        team_a = np.pad(team_a, ((0, 0), (0, -pad_size), (0, 0)), constant_values=0)
    else:
        team_b = np.pad(team_b, ((0, 0), (0, pad_size), (0, 0)), constant_values=0)

    return team_a, team_b

def mirror_positions(positions, horizontal=True, vertical=False, court_width=40, court_height=20):
    """Mirrors the given positions of players and ball on the court.
    Horizontal mirroring effectively switches sides whereas vertical mirroring
    switches left and right.

    Args:
        positions (np.ndarray): Player and ball positions.
        horizontal (bool, optional): Mirror horizontally. Defaults to False.
        vertical (bool, optional): Mirror vertically. Defaults to True.
    """
    if vertical:
        positions[:, :, 1] = court_height - positions[:, :, 1]
    if horizontal:
        positions[:, :, 0] = court_width - positions[:, :, 0]

    return positions

if "__main__" == __name__:
    data = HandballSyncedDataset("/nfs/home/rhotertj/datasets/hbl/meta3d.csv", 12, sampling_rate=2)
    from utils import array2gif, draw_trajectory
    idx = 187
    instance = data[idx]
    print("instance label", instance["label"], type(instance["label"]))
    print("instance label idx", instance["label_offset"])
    print("positions", instance["positions"])
    frames = instance["frames"]
    #[C, F, H, W]
    frames = np.transpose(frames, (3, 0, 1, 2))
    positions = instance["positions"]
    array2gif(frames, f"./img/instance_{idx}.gif", 10)
    
    fig = draw_trajectory(positions)
    fig.savefig(f"./img/instance_{idx}.png")
    

