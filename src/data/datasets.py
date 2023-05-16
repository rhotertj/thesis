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
from tqdm import tqdm

from data.data_utils import (
    get_index_offset,
    check_label_within_slice,
    combine_teams_with_indicator,
    mirror_positions,
    ensure_correct_team_size,
    combine_ball_with_indicator,
    PositionContainer
)


class MultiModalHblDataset(Dataset):

    def __init__(
        self,
        meta_path: str,
        seq_len: int = 16,
        sampling_rate: int = 1,
        load_frames: bool = True,
        transforms: Union[None, Compose] = None,
        label_mapping: callable = lambda x: x,
        overlap: bool = True
    ):
        """
        This dataset provides a number of video frames (determined by seq_len) and
        corresponding positional data for ball and players.
        If available, it also provides action label and its position in the current frame stack.

        Note that by increasing the sampling rate, the temporal range of the sampled frames increases,
        whereas the sequence length stays fixed.

        Args:
            meta_path (str): Path to the .csv file that holds paths to frames, annotations and positions.
            seq_len (int, optional): An even desired number of frames. Defaults to 16.
            sampling_rate (int, optional): Sample every nth frame. When set to 1, we sample subsequent frames
                corresponding to seq_len. Defaults to 1.
            load_frames (bool, optional): Whether to read and process images. Defaults to True.
            transforms (Union[None, Compose], optional): Transforms to apply to the video frames. Defaults to None.
            label_mapping (callable optional): A function that maps the label dictionary to an integer. 
                Defaults to the identity function.
            overlap (bool, optional): Whether to sequences may overlap in frames or not. Defaults to True.
        """
        super(MultiModalHblDataset).__init__()

        self.seq_len = (seq_len // 2) * 2
        self.seq_half = seq_len // 2
        self.load_frames = load_frames
        self.sampling_rate = sampling_rate
        self.transforms = transforms
        self.label_mapping = label_mapping
        self.overlap = overlap

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
        if not self.overlap:
            return (self.index_tracker[-1] // (self.seq_len * self.sampling_rate)) - 1

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

        Note that the provided index is continuous over the whole dataset.
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

        if not self.overlap:
            idx *= (self.seq_half * self.sampling_rate)

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
        label, label_offset, label_idx = check_label_within_slice(frame_idx - half_range, frame_idx + half_range, events, frame_idx, self.sampling_rate)
        label = self.label_mapping(label)

        for window_idx in range(frame_idx - half_range, frame_idx + half_range, self.sampling_rate):
            window_indices.append(window_idx)

            if self.load_frames:
                frame_path = str(frame_base_path / f"{str(window_idx).rjust(6, '0')}.jpg")
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        if self.load_frames:
            frames = np.stack(frames)


        team_a_pos, team_b_pos, ball_pos, _ = self.position_arrays[match_number]

        position_slice = slice(
            (frame_idx - half_range) - positions_offset,
            (frame_idx + half_range) - positions_offset,
            self.sampling_rate,
        )
        team_a_pos = team_a_pos[position_slice]
        team_b_pos = team_b_pos[position_slice]
        ball_pos = ball_pos[position_slice]

        positions = PositionContainer(
            team_a=team_a_pos,
            team_b=team_b_pos,
            ball=ball_pos,
            mirror_horizontal=self.mirror_horizontal[match_number],
            mirror_vertical=self.mirror_vertical[match_number],
        )

        if self.transforms:
            frames, positions = self.transforms(
                {"frames" : frames, "positions" : positions}
            ).values()

        instance = {
            "frames": frames,
            "positions": positions,
            "label": label,
            "label_offset": label_offset,
            "frame_idx": frame_idx,
            "query_idx": idx,
            "label_idx" : label_idx,
            "window_indices": window_indices,
            "match_number": match_number,
        }
        return instance

    def get_class_proportions(self):
        if not isinstance(self.__getitem__(0)["label"], int):
            raise TypeError("Expected Integer label. To use this function, please pass a label_mapping")
        old_setting = self.load_frames
        self.load_frames = False
        labels = Counter()
        for i in tqdm(range(len(self)), total=len(self)):
            l = self.__getitem__(i)["label"]
            labels[l] += 1
        self.load_frames = old_setting
        return labels

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


class ResampledHblDataset(Dataset):

    def __init__(
        self,
        meta_path: str,
        idx_to_frame: str,
        seq_len: int = 16,
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
            idx_to_frame (str): Path to a .jsonl file that holds an alternative dataset size and reduced number of frames.
                This allows for a stratified class frequency.
            seq_len (int, optional): An even desired number of frames. Defaults to 16.
            sampling_rate (int, optional): Sample every nth frame. When set to 1, we sample subsequent frames
                corresponding to seq_len. Defaults to 1.
            load_frames (bool, optional): Whether to read and process images. Defaults to True.
            transforms (Union[None, Compose], optional): Transforms to apply to the video frames. Defaults to None.
            label_mapping (callable, optional): A function that maps the label dictionary to an integer. 
                Defaults to the identity function.
        """
        super().__init__()

        self.seq_len = (seq_len // 2) * 2
        self.seq_half = seq_len // 2
        self.load_frames = load_frames
        self.sampling_rate = sampling_rate
        self.transforms = transforms
        self.label_mapping = label_mapping

        print("Read", meta_path)
        self.meta_df = pd.read_csv(meta_path)
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

        self.idx_to_frame_number = pd.read_json(idx_to_frame, lines=True)

    def __len__(self):
        """Length of the dataset over all matches.

            Sliding window approach needs a full sequence as padding,
            half in the beginning and half at the end.

        Returns:
            int: Dataset length.
        """
        return len(self.idx_to_frame_number)

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
            instance = self.idx_to_frame_number.iloc[idx]
            match_number, frame_idx = instance["match_number"], instance["frame_idx"]
            # NOTE: jsonl file already has buffed frame idx
            # frame_idx += (self.seq_half * self.sampling_rate)

        frame_base_path = Path(self.frame_paths[match_number])
        events = self.event_dfs[match_number]
        frames = []
        window_indices = []  # debug information

        label = {}  # Default to 'background' action
        label_offset = 0  # Which frame of the window is portraying the action

        half_range = self.seq_half * self.sampling_rate
        label, label_offset, label_idx = check_label_within_slice(frame_idx - half_range, frame_idx + half_range, events, frame_idx, self.sampling_rate)
        label = self.label_mapping(label)

        # Iterate over window, load frames and check for event
        for window_idx in range(frame_idx - half_range, frame_idx + half_range, self.sampling_rate):
            window_indices.append(window_idx)
            
            if self.load_frames:
                frame_path = str(frame_base_path / f"{str(window_idx).rjust(6, '0')}.jpg")
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        if self.load_frames:
            frames = np.stack(frames)

        team_a_pos, team_b_pos, ball_pos, _ = self.position_arrays[match_number]

        position_slice = slice(
            (frame_idx - half_range) - positions_offset,
            (frame_idx + half_range) - positions_offset,
            self.sampling_rate,
        )
        team_a_pos = team_a_pos[position_slice]
        team_b_pos = team_b_pos[position_slice]
        ball_pos = ball_pos[position_slice]

        positions = PositionContainer(
            team_a=team_a_pos,
            team_b=team_b_pos,
            ball=ball_pos,
            mirror_horizontal=self.mirror_horizontal[match_number],
            mirror_vertical=self.mirror_vertical[match_number],
        )

        if self.transforms:
            frames, positions = self.transforms(
                {"frames" : frames, "positions" : positions}
            ).values()

        instance = {
            "frames": frames,
            "positions": positions,
            "label": label,
            "label_offset": label_offset,
            "frame_idx": frame_idx,
            "label_idx" : label_idx,
            "query_idx": idx,
            "window_indices": window_indices,
            "match_number": match_number
        }
        return instance

    def get_class_proportions(self):
        if not isinstance(self.__getitem__(0)["label"], int):
            raise TypeError("Expected Integer label. To use this function, please pass a label_mapping")
        old_setting = self.load_frames
        self.load_frames = False
        labels = Counter()
        for i in tqdm(range(len(self)), total=len(self)):
            l = self.__getitem__(i)["label"]
            labels[l] += 1
        self.load_frames = old_setting
        print("Load Frames after proportions", self.load_frames)
        print("Labels", labels)
        print("Value counts:", self.idx_to_frame_number.value_counts(["class_coarse"]))
        return labels

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


if "__main__" == __name__:
    data = MultiModalHblDataset(
        "/nfs/home/rhotertj/datasets/hbl/meta3d_train.csv",
        seq_len=16,
        sampling_rate=2,
        load_frames=True,
    )
    idx = 186319
    instance = data[idx]

    print("All good")
    exit()
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
