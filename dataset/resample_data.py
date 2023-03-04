import os
import sys
from pathlib import Path
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

sys.path.append("/nfs/home/rhotertj/Code/thesis/src")
from utils import generate_class_description_fine, generate_class_description_coarse
from data.datasets import MultiModalHblDataset, ResampledHblDataset


def create_dataframe_from_dataset(dataset: Dataset, path: str = None) -> pd.DataFrame:
    """Creates a `pd.DataFrame` that contains information about frame number, match and label per instance.

    Args:
        dataset (_type_): The dataset.
        path (str, optional): A path to save the dataframe as a jsonl file. Defaults to None.

    Returns:
        pd.DataFrame: Dataset DataFrame.
    """
    data_dict = []
    for idx in tqdm(range(len(dataset))):
        instance = dataset[idx]
        label = instance["label"]
        data_dict.append(
            {
                "idx": idx,
                "frame_idx": instance["frame_idx"],
                "label": label,
                "shot": label.get("Wurf"),
                "pass": label.get("Pass"),
                "outcome": label.get("Ergebnis"),
                "body": label.get("Körper"),
                "team": label.get("Team"),
                "hand": label.get("Hand"),
                "match_number": instance["match_number"]
            }
        )
    df = pd.DataFrame(data_dict)
    df = df.replace({np.nan: None})

    df["class_fine"] = df.apply(generate_class_description_fine, axis=1)
    df["class_coarse"] = df["class_fine"].apply(generate_class_description_coarse)
    if path:
        print("Save dataset to", path)
        df.to_json(path, lines=True, orient="records")
    return df


def random_split_dataframe(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """Random splits a `pd.DataFrame` into train, val and test split.

    Args:
        df (pd.DataFrame): DataFrame to split.
        val_size (float, optional): Size of the validation split. Defaults to 0.15.
        test_size (float, optional): Size of the test split. Defaults to 0.15.

    Returns:
        Tuple[List[int]]: Indices of the splits.
    """
    train_size = 1 - test_size
    df_idx = df.index.tolist()  # serves as X
    y = df["class_coarse"]
    train_idx, test_idx, *_ = train_test_split(df_idx, stratify=y, test_size=test_size)
    y = df.loc[train_idx, "class_coarse"]
    train_idx, val_idx, *_ = train_test_split(train_idx, stratify=y, test_size=val_size / train_size)
    return train_idx, val_idx, test_idx


def time_split_dataframe(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """Splits a `pd.DataFrame` into train, val and test split.

    Args:
        df (pd.DataFrame): DataFrame to split.
        val_size (float, optional): Size of the validation split. Defaults to 0.15.
        test_size (float, optional): Size of the test split. Defaults to 0.15.

    Returns:
        Tuple[List[int]]: Indices of the splits.
    """
    train_size = 1 - test_size
    df_idx = df.index.tolist()
    # shuffle=False ensures splitting along temporal axis
    train_idx, test_idx, *_ = train_test_split(df_idx, shuffle=False, test_size=test_size)
    train_idx, val_idx, *_ = train_test_split(train_idx, shuffle=False, test_size=val_size / train_size)
    return train_idx, val_idx, test_idx


def load_split_matches_df(data_path: Path, sequence_length: int, sampling_rate: int, overlap: bool, balanced: bool):

    overlaps = [overlap]
    if balanced:
        overlaps = (True, False)

    dfs = []
    for _overlap in overlaps:
        trainset = MultiModalHblDataset(
            meta_path=data_path / "meta3d_train.csv",
            seq_len=sequence_length,
            sampling_rate=sampling_rate,
            load_frames=False,
            overlap=_overlap
        )
        valset = MultiModalHblDataset(
            meta_path=data_path / "meta3d_val.csv",
            seq_len=sequence_length,
            sampling_rate=sampling_rate,
            load_frames=False,
            overlap=_overlap
        )
        testset = MultiModalHblDataset(
            meta_path=data_path / "meta3d_test.csv",
            seq_len=sequence_length,
            sampling_rate=sampling_rate,
            load_frames=False,
            overlap=_overlap
        )

        train_df = create_dataframe_from_dataset(trainset)
        val_df = create_dataframe_from_dataset(valset)
        test_df = create_dataframe_from_dataset(testset)
        dfs.append((train_df, val_df, test_df))

    return dfs


def balance_classes(overlap_df: pd.DataFrame, chunk_df: pd.DataFrame) -> pd.DataFrame:
    # TODO Size of background class
    overlapped_shots = overlap_df[overlap_df.class_coarse == "Shot"]
    overlapped_passes = overlap_df[overlap_df.class_coarse == "Pass"].iloc[:len(overlapped_shots)]
    simple_background = chunk_df[chunk_df.shot.isnull()]
    balanced_df = pd.concat([overlapped_passes, simple_background, overlapped_shots])
    balanced_df.sort_values(by=["frame_idx"], inplace=True)
    balanced_df.reset_index(inplace=True, drop=True)
    return balanced_df


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Resample the dataset!')
    parser.add_argument(
        "-d", "--data_path", type=str, help='Path for meta data.', default=Path("/nfs/home/rhotertj/datasets/hbl")
    )
    parser.add_argument('-m', "--mode", type=str, help='Splitting criterion.', choices=['matches', 'random', 'time'])
    parser.add_argument('-v', "--val_size", type=float, help="Split size of validation set.")
    parser.add_argument('-t', "--test_size", type=float, help="Split size of test set.")
    parser.add_argument(
        '-b', "--balanced", type=bool, help="Whether upsampling of underrepresented classes takes place.", default=False
    )
    parser.add_argument(
        '-o',
        '--overlap',
        type=bool,
        help="Whether to use overlapping sliding windows. Ignored when --balanced is True.",
        default=True
    )
    parser.add_argument('-l', '--sequence_length', type=int, help="Sequence length.")
    parser.add_argument('-r', '--sampling_rate', type=int, help="Sampling rate.")

    args = parser.parse_args()

    if args.mode == "random" and (args.overlap == True or args.balanced == True):
        print(
            "Please do not use random splits with overlapping sliding windows. This will lead to the same event landing in more than one split."
        )
        exit(1)


    fpath = args.data_path / "resampled" / "balanced" / str(args.balanced) / "overlap" / str(args.overlap) / "sql_sr" / f"{args.sequence_length}x{args.sampling_rate}" / "mode" / args.mode
    print("Creating", fpath)
    os.makedirs(
        fpath,
        exist_ok=True
    )

    if args.mode == 'matches':
        split_match_dfs = load_split_matches_df(
            args.data_path, args.sequence_length, args.sampling_rate, args.overlap, args.balanced
        )
        splits = {}
        if args.balanced:
            overlapped_dfs, chunk_dfs = split_match_dfs
            for split, overlap_df, chunk_df in zip(["train", "val", "test"], overlapped_dfs, chunk_dfs):
                split_df = balance_classes(overlap_df, chunk_df)
                splits[split] = split_df

        else:
            for split, df in zip(["train", "val", "test"], split_match_dfs[0]):
                splits[split] = df

        for split, df in splits.items():
            fname = fpath / f"meta3d_{split}.jsonl"
            print(df.value_counts(["class_coarse"]))
            print("Writing to", fname)
            df.to_json(fname, lines=True, orient="records")

    if args.mode in ('time', 'random'):
        overlaps = [args.overlap]
        if args.balanced:
            overlaps = (True, False)

        dfs = []
        for _overlap in overlaps:
            dataset = MultiModalHblDataset(
                meta_path=args.data_path / "meta3d.csv",
                seq_len=args.sequence_length,
                sampling_rate=args.sampling_rate,
                load_frames=False,
                overlap=_overlap
            )

        if args.balanced:
            overlap_df, chunk_df = dfs
            balanced_df = balance_classes(overlap_df, chunk_df)
            dfs = [balanced_df]

        if args.mode == "time":
            train_idx, val_idx, test_idx = time_split_dataframe(dfs[0], args.val_size, args.test_size)

        if args.mode == "random":
            train_idx, val_idx, test_idx = random_split_dataframe(dfs[0], args.val_size, args.test_size)

        for split, idx in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
            fname = fpath / f"meta3d_{split}.jsonl"
            split_df = dfs[0].iloc[idx]
            split_df.reset_index(inplace=True, drop=True)
            print(df.value_counts(["class_coarse"]))
            print("Writing to", fname)
            split_df.to_json(fname, lines=True, orient="records")
