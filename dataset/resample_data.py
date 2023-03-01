import os
import sys
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

sys.path.append("/nfs/home/rhotertj/Code/thesis/src")
from utils import generate_class_description_fine, generate_class_description_coarse
from data.datasets import MultiModalHblDataset, ResampledHblDataset

def create_dataframe_from_dataset(dataset : Dataset, path : str = None) -> pd.DataFrame:
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
                "idx" : idx,
                "frame_idx" : instance["frame_idx"],
                "label" : label,
                "shot" : label.get("Wurf"),
                "pass" : label.get("Pass"),
                "outcome" : label.get("Ergebnis"),
                "body" : label.get("Körper"),
                "team" : label.get("Team"),
                "hand" : label.get("Hand"),
                "match_number" : instance["match_number"]
            }
        )
    df = pd.DataFrame(data_dict)
    df = df.replace({np.nan:None})

    df["class_fine"] = df.apply(generate_class_description_fine, axis=1)
    df["class_coarse"] = df["class_fine"].apply(generate_class_description_coarse)
    if path:
        print("Save dataset to", path)
        df.to_json(path, lines=True, orient="records")
    return df

def random_split_dataframe(df : pd.DataFrame, val_size : float = 0.15, test_size : float = 0.15):
    """Random splits a `pd.DataFrame` into train, val and test split.

    Args:
        df (pd.DataFrame): DataFrame to split.
        val_size (float, optional): Size of the validation split. Defaults to 0.15.
        test_size (float, optional): Size of the test split. Defaults to 0.15.

    Returns:
        Tuple[List[int]]: Indices of the splits.
    """    
    train_size = 1 - test_size
    df_idx = df.index.tolist() # serves as X
    y = df["class_coarse"]
    train_idx, test_idx, *_ = train_test_split(df_idx, stratify=y, test_size=test_size)
    y = df.loc[train_idx, "class_coarse"]
    train_idx, val_idx, *_ = train_test_split(train_idx, stratify=y, test_size=val_size/train_size)
    return train_idx, val_idx, test_idx


def split_dataframe(df : pd.DataFrame, val_size : float = 0.15, test_size : float = 0.15):
    """Random splits a `pd.DataFrame` into train, val and test split.

    Args:
        df (pd.DataFrame): DataFrame to split.
        val_size (float, optional): Size of the validation split. Defaults to 0.15.
        test_size (float, optional): Size of the test split. Defaults to 0.15.

    Returns:
        Tuple[List[int]]: Indices of the splits.
    """    
    train_size = 1 - test_size
    df_idx = df.index.tolist()
    train_idx, test_idx, *_ = train_test_split(df_idx, shuffle=False, test_size=test_size)
    train_idx, val_idx, *_ = train_test_split(train_idx, shuffle=False, test_size=val_size/train_size)
    return train_idx, val_idx, test_idx



if "__main__" == __name__:
    # TODO Go for argparse with defaults
    data_path = Path("/nfs/home/rhotertj/datasets/hbl")
    sql = 16
    sr = 2

    truefalse = [True, False]
    dfs = []
    for overlap in truefalse:
        filename = f"meta3d_{sql}_{sr}_{'overlap' if overlap else 'nooverlap'}.jsonl"        
        if not os.path.exists(data_path / filename):
            dataset = MultiModalHblDataset(
            meta_path=data_path / "meta3d.csv",
            seq_len=sql,
            sampling_rate=sr, 
            load_frames=False,
            overlap=overlap
            )
            df = create_dataframe_from_dataset(dataset, data_path / filename)
        else:
            print("Load dataframe from", filename)
            df = pd.read_json(data_path / filename, lines=True)
        dfs.append(df)

    overlap_df, non_overlap_df = dfs

    overlapped_shots = overlap_df[overlap_df.class_coarse == "Shot"]
    simple_passes = overlap_df[overlap_df.class_coarse == "Pass"].iloc[:len(overlapped_shots)]
    simple_background = non_overlap_df[non_overlap_df.shot.isnull()]
    full_df = pd.concat([simple_passes, simple_background, overlapped_shots])
    full_df.sort_values(by=["frame_idx"], inplace=True)
    full_df.reset_index(inplace=True, drop=True)

    # print(full_df.value_counts(["class_coarse"]))
    train_idx, val_idx, test_idx = split_dataframe(full_df)

    train_df = full_df.iloc[train_idx]
    train_df.reset_index(inplace=True, drop=True)
    # print(train_df.value_counts(["class_coarse"]))
    train_df.to_json(data_path / f"meta3d_{sql}_{sr}_train_balanced.jsonl", lines=True, orient="records")

    val_df = full_df.iloc[val_idx]
    val_df.reset_index(inplace=True, drop=True)
    print(val_df.value_counts(["class_coarse"]))
    val_df.to_json(data_path / f"meta3d_{sql}_{sr}_val_balanced.jsonl", lines=True, orient="records")


    test_df = full_df.iloc[test_idx]
    test_df.reset_index(inplace=True, drop=True)
    # print(test_df.value_counts(["class_coarse"]))
    test_df.to_json(data_path / f"meta3d_{sql}_{sr}_test_balanced.jsonl", lines=True, orient="records")


    # TODO: Why is the distribution of classes different when iterating over the dataset than in the source dataframe / jsonl?
    from data.labels import LabelDecoder
    ld = LabelDecoder(3)
    dataset = ResampledHblDataset(
        idx_to_frame=data_path / f"meta3d_{sql}_{sr}_val_balanced.jsonl",
        meta_path=data_path / "meta3d.csv",
        seq_len=sql,
        sampling_rate=sr, 
        load_frames=False,
        label_mapping=ld,
    )
    print(dataset.get_class_proportions())

    for i, instance in tqdm(enumerate(dataset)):
        assert i == instance["query_idx"]
        # frame is different, maybe due to window movement?
        assert val_df.loc[i, "frame_idx"] == instance["frame_idx"], f"{(val_df.loc[i, 'frame_idx'], instance['frame_idx'])}"
        assert ld(val_df.loc[i, "label"]) == instance["label"], f"{ld(val_df.loc[i, 'label']), instance['label']}"