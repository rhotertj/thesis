"""
This code regarding the synchronization of positions and video frames 
is adapted from the single-match-example

    -> /nfs/data/mm4spa/mm_hbl/scripts/sync/single_match_example.py

This script is intended to synchronize position and video data and their respective annotations.
"""

import sys

sys.path.append("/nfs/data/mm4spa/mm_hbl/scripts/sync/")
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle as pkl
from tqdm import tqdm
import scipy
from dataclasses import dataclass
from typing import Union
import sync
from kinexon_reader import read_kinexon_file

DATA_SOURCE = Path("/nfs/data/mm4spa/mm_hbl/hbl_19-20")

EVENTS_PATH = Path("/nfs/home/rhotertj/datasets/hbl/events")
POSITIONS_PATH = Path("/nfs/home/rhotertj/datasets/hbl/positions")
FRAMES_PATH = Path("/data/hbl_rawframes/")
VIDEO_PATH = Path("/nfs/home/rhotertj/datasets/hbl/videos")
META_PATH = Path("/nfs/home/rhotertj/datasets/hbl/")

@dataclass
class SyncInformation:
    match_id_min: Union[str, int] # SportRadar match_id without "sr:sport_event:"
    raw_pos_knx: str  # kinexon pos filename
    raw_video: str  # video filename
    v_h1_start: float  # t_v [s]
    v_h1_end: float  # t_v [s]
    v_h2_start: float  # t_v [s]
    v_h2_end: float  # t_v [s]
    p_h1: float  # t_v [s]
    p_h2: float  # t_v [s]
    duration: float  # t_v [s]
    avg_frame_rate: float  # average frame rate video
    avg_frame_rate_pos: float  # average frame rate position data
    t_null: float  # [ms] event timestamp (local time) of first recorded position data
    mirror_horizontal : bool
    mirror_vertical: bool
    split: str

def copy_and_rename_videos(meta_df):
    for event_id, row in meta_df.iterrows():
        src = DATA_SOURCE / "raw_video" / row["raw_video"]
        target = VIDEO_PATH / f"{event_id.replace('sr:sport_event:', '')}.mp4"
        shutil.copy(src, target)


def read_all_matches_meta():
    # base mapping to video and position data
    index_col = "match_id"
    usecols = [index_col, "raw_pos_knx", "raw_video"]
    df_matches = pd.read_csv(DATA_SOURCE / "raw_mapping.csv", index_col=index_col, usecols=usecols)

    # offsets necessary for sync
    usecols = [
        index_col,
        "v_h1_start",
        "v_h1_end",
        "v_h2_start",
        "v_h2_end",
        "p_h1",
        "p_h2",
        "mirror_horizontal",
        "mirror_vertical",
        "split"
    ]
    # df_anno = pd.read_csv(DATA_SOURCE / "raw_base_annotations.csv", index_col=index_col, usecols=usecols)
    df_anno = pd.read_csv("/nfs/home/rhotertj/datasets/hbl/raw_base_annotations.csv", index_col=index_col, usecols=usecols)

    df = df_matches.join(df_anno)
    df = df.dropna(axis=0, subset=["v_h1_start", "p_h1"])

    # necessary video information
    usecols = [index_col, "duration", "avg_frame_rate"]
    df_video = pd.read_csv(DATA_SOURCE / "raw_video.csv", index_col=index_col, usecols=usecols)
    df = df.join(df_video)
    return df


def read_sync_sportradar_data(fname, meta_info):
    with open(fname) as fr:
        # dict_keys(['generated_at', 'sport_event', 'sport_event_status', 'statistics', 'timeline'])
        json_events = json.load(fr)
    df_events = pd.DataFrame.from_records(json_events["timeline"]).sort_values("time")

    df_events["localtime_ms"] = pd.to_datetime(df_events["time"], unit="ns").view(np.int64) / int(1e6)

    df_events = df_events.loc[df_events["type"].isin(
        ["period_start", "break_start", "match_started", "match_ended", "shot_blocked"]
    )]
    df_events = df_events[["type", "time", "localtime_ms"]]
    df_events["t_v"] = sync.localtime2video(
        t_w=pd.to_datetime(df_events["localtime_ms"], unit="ms"),
        t_null=pd.to_datetime(meta_info.t_null, unit="ms"),
        v_h1_start=meta_info.v_h1_start,
        v_h2_start=meta_info.v_h2_start,
        offset_p_h1=meta_info.p_h1,
        offset_p_h2=meta_info.p_h2,
    )  # aligned video timestamp t_v in [s]
    return df_events


def resample(dxy: np.ndarray, resample_factor: float) -> np.ndarray:
    """Resample dim=0 and interpolate each element of dim=1 individually.

    Args:
        dxy (np.ndarray): ndarray of [T, N, ?] where T is modified and the last dimension is interpolated for each N
        resample_factor (float): resampling factor

    Returns:
        np.ndarray: ndarray of shape [T*resample_factor, N, ?]
    """
    t_r = int(dxy.shape[0] * resample_factor)  # new target time dimension

    if dxy.ndim == 2:  # ball positions, no player dimension
        dxy = np.expand_dims(dxy, 1)

    def _resample(x: np.ndarray):
        # From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
        # "Calling interp1d with NaNs present in input values results in undefined behaviour."
        interp_fn = scipy.interpolate.interp1d(range(len(x)), x)
        xr = interp_fn(np.linspace(0, len(x) - 1, num=t_r))
        return xr

    drxy = np.zeros((t_r, *dxy.shape[1:]), dtype=np.float64)

    for player in range(dxy.shape[-2]):
        for dim in range(dxy.shape[-1]):
            drxy[:, player, dim] = _resample(dxy[:, player, dim])
    return drxy


def align_position_to_video(pos_sets, meta_info):
    # Resample pos to video frame rate by interpolation
    resampling_factor = meta_info.avg_frame_rate / meta_info.avg_frame_rate_pos
    pos_sets = [resample(xyz, resampling_factor) for xyz in pos_sets]
    # print(f"After resampling to {meta_info.avg_frame_rate=}fps", [x.shape for x in pos_sets])

    # Align position data to video
    pos_sets = [
        sync.pos2video(
            pos,
            frames_video=int(round(meta_info.duration * meta_info.avg_frame_rate)),
            v_h1_start=int(round(meta_info.v_h1_start * meta_info.avg_frame_rate)),
            v_h1_end=int(round(meta_info.v_h1_end * meta_info.avg_frame_rate)),
            v_h2_start=int(round(meta_info.v_h2_start * meta_info.avg_frame_rate)),
            v_h2_end=int(round(meta_info.v_h2_end * meta_info.avg_frame_rate)),
            offset_p_h1=int(round(meta_info.p_h1 * meta_info.avg_frame_rate)),
            offset_p_h2=int(round(meta_info.p_h2 * meta_info.avg_frame_rate)),
            constant_values=0.0,
            replace_nan=0.0,
        ) for pos in pos_sets
    ]
    # print(f"After alignment with video num_frames_video={int(round(meta_info.duration * meta_info.avg_frame_rate))}:", [x.shape for x in pos_sets])
    return pos_sets


def main():

    # Get annotation filenames for given matches
    event2json_df = pd.read_csv("/nfs/data/mm4spa/mm_hbl/hbl_19-20/events_its.csv", index_col="match_id")
    match_id2event_json = event2json_df.to_dict()["events_its"]

    # Read meta info for all matches, including their video and position files
    meta_df = read_all_matches_meta()

    new_meta_df = meta_df.copy(deep=True)
    new_meta_df["n_frames"] = round(meta_df["duration"] * meta_df["avg_frame_rate"])
    new_meta_df["frames_path"] = ""
    new_meta_df["events_path"] = ""
    new_meta_df["positions_path"] = ""

    for match_number in tqdm(range(len(meta_df))):
        # Prepare meta info for selected match
        meta_info = SyncInformation(
            **{
                **meta_df.iloc[match_number].to_dict(),
                "match_id_min":
                    meta_df.index[match_number].replace("sr:sport_event:", ""),
            },
            avg_frame_rate_pos=None,
            t_null=None,
        )
        # Read position data (frame rate != video frame rate) and resample positions to video frame rate
        # Then, align position data such that each frame matches one position
        # print("Reading info for", meta_info.match_id_min)
        pos_sets, t_start, avg_frame_rate_pos, timestamps = \
            read_kinexon_file(
            DATA_SOURCE / "raw_pos_knx" / meta_info.raw_pos_knx
            )

        meta_info.t_null = t_start
        meta_info.avg_frame_rate_pos = avg_frame_rate_pos
        # print(f"{meta_info.avg_frame_rate_pos=}fps", [x.shape for x in pos_sets])

        pos_sets = align_position_to_video(pos_sets, meta_info)

        pos_available = sync.pos_available(pos_sets[0], pos_sets[1], value_check="zeros")
        pos_sets.append(pos_available)

        # legacy from time where only 10 matches were available
        # vertical = [1, 2, 5, 6, 8, 10]
        # horizontal = [1, 2, 5, 6, 8, 10]

        # new_meta_df.loc[meta_df.index[match_number], "mirror_vertical"] = match_number in vertical
        # new_meta_df.loc[meta_df.index[match_number], "mirror_horizontal"] = match_number in horizontal

        # train = [0, 1, 3, 6, 9, 10]
        # val = [2, 7]
        # test = [8, 4]

        # if match_number in train:
        #     stage = "train"
        # elif match_number in val:
        #     stage = "val"
        # elif match_number in test:
        #     stage = "test"

        # new_meta_df.loc[meta_df.index[match_number], "stage"] = stage

        event_file = match_id2event_json[f"sr:sport_event:{meta_info.match_id_min}"]
        # print("Read", DATA_SOURCE / "events_its" / event_file)
        events_df = pd.read_json(DATA_SOURCE / "events_its" / event_file, lines=True)

        events_path = EVENTS_PATH / (meta_info.match_id_min + ".csv")
        events_df.to_csv(events_path)
        new_meta_df.loc[meta_df.index[match_number], "events_path"] = events_path

        positions_path = POSITIONS_PATH / (meta_info.match_id_min + "3d.npy")
        with open(positions_path, "wb+") as f:
            pkl.dump(pos_sets, f)
        new_meta_df.loc[meta_df.index[match_number], "positions_path"] = positions_path

        new_meta_df.loc[meta_df.index[match_number], "frames_path"] = FRAMES_PATH / f"{meta_info.match_id_min}.mp4.d"

    new_meta_df.to_csv(META_PATH / "meta30.csv")
    for split in ["train", "val", "test"]:
        new_meta_df[new_meta_df.split == split].to_csv(META_PATH / f"meta30_{split}.csv")
    print(new_meta_df)


if __name__ == "__main__":
    main()