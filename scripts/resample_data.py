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
from data.datasets import MultiModalHblDataset, ResampledHblDataset

META_FILE = "meta30"

def create_dataframe_from_dataset(dataset: Dataset, path: str = None) -> pd.DataFrame:
    """Creates a `pd.DataFrame` that contains information about frame number, match and label per instance.

    Args:
        dataset (_type_): The dataset.
        path (str, optional): A path to save the dataframe as a jsonl file. Defaults to None.

    Returns:
        pd.DataFrame: Dataset DataFrame.
    """
    data_dict = []
    print("Creating dataframe from dataset...")
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
    print("Returning dataset of length", len(df))
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


def load_split_matches_df(data_path: Path, sequence_length: int, sampling_rate: int, overlap: bool):

    trainset = MultiModalHblDataset(
        meta_path=data_path / f"{META_FILE}_train.csv",
        seq_len=sequence_length,
        sampling_rate=sampling_rate,
        load_frames=False,
        overlap=overlap
    )
    valset = MultiModalHblDataset(
        meta_path=data_path / f"{META_FILE}_valid.csv",
        seq_len=sequence_length,
        sampling_rate=sampling_rate,
        load_frames=False,
        overlap=overlap
    )
    testset = MultiModalHblDataset(
        meta_path=data_path / f"{META_FILE}_test.csv",
        seq_len=sequence_length,
        sampling_rate=sampling_rate,
        load_frames=False,
        overlap=overlap
    )

    train_df = create_dataframe_from_dataset(trainset)
    val_df = create_dataframe_from_dataset(valset)
    test_df = create_dataframe_from_dataset(testset)

    return train_df, val_df, test_df


def balance_classes(df: pd.DataFrame, background_size: float, upsample=True) -> pd.DataFrame:
    if upsample:
        overlapped_passes = df[df.class_coarse == "Pass"]
        overlapped_shots = df[df.class_coarse == "Shot"].sample(len(overlapped_passes), replace=True)
    
    else:
        overlapped_shots = df[df.class_coarse == "Shot"]
        overlapped_passes = df[df.class_coarse == "Pass"].iloc[:len(overlapped_shots)]
    
    n_shots_passes = len(overlapped_passes) + len(overlapped_shots)
    frac_shots_passes = 1 - background_size
    n_background = int((n_shots_passes / frac_shots_passes) * background_size)

    print("Sampling n background", n_background, "for shots and passes:", n_shots_passes)
    overlapped_background = df[df.shot.isnull()].sample(n_background, replace=n_background > len(df[df.shot.isnull()]))
    balanced_df = pd.concat([overlapped_passes, overlapped_background, overlapped_shots])
    balanced_df.sort_values(by=["frame_idx"], inplace=True)
    balanced_df.reset_index(inplace=True, drop=True)
    return balanced_df


def generate_class_description_fine(row):
    pass_technique = {
        "O":  None,
        "A": "Schlagwurfpass",
        "B": "Handgelenkspass",
        "C": "Druckpass",
        "D": "Rückhandpass",
        "E": "Undefinierter Pass",
        "X":  None
    }
    shot_technique = {
        0: None,
        1: "Sprungwurf Außen",
        2: "Sprungwurf Rückraum",
        3: "Sprungwurf Torraum/Zentrum",
        4: "Standwurf mit Anlauf",
        5: "Standwurf ohne Anlauf",
        6: "Drehwurf",
        7: "7-Meter",
        8: "Undefinierter Wurf", 
    }
    if row["shot"] == None:
        return "Background"
    elif row["shot"] == "0" and row["pass"] in ("O", "X"):
        return "Foul"
    elif row["pass"] in ("O", "X"):
        return shot_technique[int(row["shot"])]
    elif row["shot"] == "0":
        return pass_technique[row["pass"]]
    else:
        print("error at", row["pass"], row["shot"], row["label"], type(row["pass"]), type(row["shot"]))

def generate_class_description_coarse(fine_class):
    pass_technique = {
    "O":  None,
    "A": "Schlagwurfpass",
    "B": "Handgelenkspass",
    "C": "Druckpass",
    "D": "Rückhandpass",
    "E": "Undefinierter Pass",
    "X":  None
    }
    shot_technique = {
        0: None, # zero
        1: "Sprungwurf Außen",
        2: "Sprungwurf Rückraum",
        3: "Sprungwurf Torraum/Zentrum",
        4: "Standwurf mit Anlauf",
        5: "Standwurf ohne Anlauf",
        6: "Drehwurf",
        7: "7-Meter",
        8: "Undefinierter Wurf", 
    }
    if fine_class in pass_technique.values():
        return "Pass"
    if fine_class in shot_technique.values():
        return "Shot"
    else:
        return fine_class # Background and Foul



if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Resample the dataset!')
    parser.add_argument(
        "-d", "--data_path", type=str, help='Path for meta data.', default=Path("/nfs/home/rhotertj/datasets/hbl")
    )
    parser.add_argument("--mode", type=str, help='Splitting criterion.', choices=['matches', 'random', 'time'])
    parser.add_argument("--val_size", type=float, help="Split size of validation set. Should be between 0 and 1, default is 0.15. Ignored when split is `matches`.", default=0.15)
    parser.add_argument("--test_size", type=float, help="Split size of test set. Should be between 0 and 1, default is 0.15.  Ignored when split is `matches`.", default=0.15)
    parser.add_argument("--background_size", type=float, default=0.4, help="Proportion of the background class when balancing proportions. Default is 0.4.")
    parser.add_argument(
        "--balanced", help="Whether balancing of underrepresented classes takes place.", default=False, action='store_true'
    )
    parser.add_argument(
        '--overlap',
        help="Whether to use overlapping sliding windows.",
        default=False,
        action='store_true'
    )
    parser.add_argument('--sequence_length', type=int, help="Sequence length.")
    parser.add_argument('--sampling_rate', type=int, help="Sampling rate.")
    parser.add_argument('--upsample', action='store_true', help="Whether to upsample shots or downsample passes.")

    args = parser.parse_args()
    print(args)

    if args.mode == "random" and (args.overlap == True or args.balanced == True):
        print(
            "Please do not use random splits with overlapping sliding windows. This will lead to the same event landing in more than one split."
        )
        exit(1)


    fpath = args.data_path / "resampled" / "balanced" / str(args.balanced) / "overlap" / str(args.overlap) / "sql_sr" / f"{args.sequence_length}x{args.sampling_rate}" / "mode" / args.mode / "upsampled" / str(args.upsample)
    print("Creating", fpath)
    os.makedirs(
        fpath,
        exist_ok=True
    )

    # Splitting by matches:
    #   Load meta_split.csv for all splits
    #   Balance each split separately
    #   Save splits separately
    if args.mode == 'matches':
        split_match_dfs = load_split_matches_df(
            args.data_path, args.sequence_length, args.sampling_rate, args.overlap
        )
        splits = {}
        if args.balanced:
            for split, df in zip(["train", "val", "test"], split_match_dfs):
                split_df = balance_classes(df, args.background_size, upsample=args.upsample)
                splits[split] = split_df

        else:
            for split, df in zip(["train", "val", "test"], split_match_dfs):
                splits[split] = df

        for split, df in splits.items():
            fname = fpath / f"{META_FILE}_{split}.jsonl"
            print(df.value_counts(["class_coarse"]))
            print("Writing to", fname)
            df.to_json(fname, lines=True, orient="records")

    # Splitting by time or random:
    #   Load complete dataset
    #   Split either randomly or along dataset by size ('time')
    if args.mode in ('time', 'random'):
       
        dataset = MultiModalHblDataset(
            meta_path=args.data_path / f"{META_FILE}.csv",
            seq_len=args.sequence_length,
            sampling_rate=args.sampling_rate,
            load_frames=False,
            overlap=args.overlap
        )

        df = create_dataframe_from_dataset(dataset)

        if args.balanced:
            df = balance_classes(df, args.background_size, args.upsample)

        if args.mode == "time":
            train_idx, val_idx, test_idx = time_split_dataframe(df, args.val_size, args.test_size)

        if args.mode == "random":
            train_idx, val_idx, test_idx = random_split_dataframe(df, args.val_size, args.test_size)

        for split, idx in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
            fname = fpath / f"{META_FILE}_{split}.jsonl"
            split_df = df.iloc[idx]
            split_df.reset_index(inplace=True, drop=True)
            print(split_df.value_counts(["class_coarse"]))
            print("Writing to", fname)
            split_df.to_json(fname, lines=True, orient="records")
