
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import pandas as pd
from pathlib import Path
import cv2

# TODO: How to elegantly split wrt matches?
# TODO: Prepare different representation for positions
        # -> On the fly plotting might be too slow

class HandballSyncedDataset(Dataset):

    def __init__(self, meta_path : str, seq_len : int = 8, load_frames : bool = True):
        super(HandballSyncedDataset).__init__()

        self.seq_len = seq_len
        self.seq_half = seq_len // 2
        self.load_frames = load_frames

        # might be useful to pass information from pytorch lightning config about dataset
        self.meta = {}

        print(f"Read {meta_path}...")
        meta_df = pd.read_csv(meta_path)
        self.index_tracker = [0]
        self.idx_to_frame_number = []
        self.frame_paths = []
        self.event_dfs = []
        self.position_arrays = []
        for _, row in meta_df.iterrows():
            self.frame_paths.append(row["frames_path"])
            self.event_dfs.append(
                pd.read_csv(row["events_path"], index_col="t_start")
            )
            with open(row["positions_path"], "rb") as f:
                self.position_arrays.append(
                    pkl.load(f)
                )
        
        # NOTE: All arrays and paths are indexed based on frame number.
        # We need to create a mapping to ensure availability of position data 
        for *_, availability in self.position_arrays:
            kernel = np.ones(self.seq_len)
            available_windows_cvn = np.convolve(availability, kernel, mode="valid")
            available_windows = np.where(available_windows_cvn == self.seq_len)[0]
            self.idx_to_frame_number.append(available_windows)
            self.index_tracker.append(int(self.index_tracker[-1] + len(available_windows)))

        print("Tracker", self.index_tracker)


    def __len__(self):
        """Length of the dataset over all matches.
            
            Sliding window approach needs a full sequence as padding,
            half in the beginning and half at the end.

        Returns:
            int: Dataset length.
        """
        return self.index_tracker[-1] - self.seq_len

    def __getitem__(self, idx):
        """This method returns 
            - stacked frames (RGB)
            - corresponding positional data for players and ball
            - the corresponding event label
            - the annotated event position in the frame stack.

        Note that the provided index is valid over the whole dataset. It can not be treated as a frame number or index for a given match.
        It first needs to be matched against a match and a frame number, based on the availability of positional data.

        Args:
            idx (int): Index of data to be retrieved.

        Returns:
            dict: A dict containing frames, positions, label and label_offset.
        """

        # Get correct match based on idx (match_number) and idx with respect to match and availability (frame_idx) 
        # from idx with respect to dataset (param: idx)

        # Add half sequence length to index to avoid underflowing dataset idx < seq_len
        match_number, frame_idx = get_index_offset(self.index_tracker, self.idx_to_frame_number, idx + self.seq_half)
        
        frame_base_path = Path(self.frame_paths[match_number])
        events = self.event_dfs[match_number]
        frames = []
        
        label = 2 # Default to 'background' action
        label_offset = 0 # Which frame of the window is portraying the action

        # Iterate over window, load frames and check for event
        for window_idx in range(frame_idx - self.seq_half, frame_idx + self.seq_half):
            if window_idx in events.index:
                label = events.loc[window_idx].labels
                label_offset = (window_idx - frame_idx) + self.seq_half
            if self.load_frames:
                frame_path = str(frame_base_path / f"{str(window_idx).rjust(6, '0')}.jpg")
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        if self.load_frames:
            frames = np.stack(frames)

        team_a_pos, team_b_pos, ball_pos, _ = self.position_arrays[match_number]
        team_a_pos, team_b_pos = ensure_equal_teamsize(team_a_pos, team_b_pos)

        # add team information
        team_a_indicator = np.zeros((*team_a_pos.shape[:2], 1))
        team_b_indicator = team_a_indicator + 1

        team_a_pos = np.concatenate([team_a_pos, team_a_indicator], axis=-1)
        team_b_pos = np.concatenate([team_b_pos, team_b_indicator], axis=-1)

        teams_pos = np.hstack([team_a_pos, team_b_pos])

        ball_pos = ball_pos[:, 0, :] # there is just one ball

        # add z dim for ball, this should be given in the future
        ball_z = np.zeros((ball_pos.shape[0], 1))
        ball_pos = np.concatenate([ball_pos, ball_z], axis=-1)
        ball_pos = np.expand_dims(ball_pos, 1)

        all_pos = np.hstack([teams_pos, ball_pos])

        instance = {
            "frames" : frames,
            "label" : label, 
            "positions" : all_pos,
            "label_offset" : label_offset
        }
        return instance

    def export_json(self, idx):
        """Generate trajectory in json format for internal visualization tool.
        """
        raise NotImplementedError()


def get_index_offset(boundaries, idx2frame, idx):
    """The dataset is indexed by idx2frame frames and positions based on sequence length - not all frames have positional data.
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
        if i <= idx and j >= idx:
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

if "__main__" == __name__:
    data = HandballSyncedDataset("/nfs/home/rhotertj/datasets/hbl/meta.csv", 8)
    print(data[10])
    

