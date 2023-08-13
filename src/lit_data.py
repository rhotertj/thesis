from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pytorch_lightning as pl
import torch
import numpy as np
import dgl
from torchvision import transforms as t
import multimodal_transforms as mmt
import pytorchvideo.transforms as ptvt

from data import MultiModalHblDataset, ResampledHblDataset, LabelDecoder, PositionContainer

# NormalizeVideo(
#     mean=[0.39449842, 0.4566527, 0.49926605],
#     std=[0.18728599, 0.21862774, 0.267905]
#),

class LitMultiModalHblDataset(pl.LightningDataModule):

    def __init__(
        self,
        meta_path_train : str,
        meta_path_val : str,
        meta_path_test : str,
        overlap: bool,
        label_mapping : LabelDecoder,
        seq_len : int = 16,
        sampling_rate : int = 2,
        load_frames : bool = True,
        batch_size : int = 1,
        epsilon : int = 7,
        mix_video : bool = True,
        mm_transforms : t.Compose = None,
        position_format : str = "graph_per_sequence",
        relative_positions: bool = False,
        team_indicator : bool = True,
        ) -> None:

        super().__init__()
        self.meta_path_train = meta_path_train
        self.meta_path_val = meta_path_val
        self.meta_path_test = meta_path_test

        self.overlap = overlap
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.load_frames = load_frames
        self.label_mapping = label_mapping
        self.batch_size = batch_size
        self.val_transforms = t.Compose([
            mmt.FrameSequenceToTensor(),
            mmt.Resize(size=(224,224))
            ])

        self.train_transforms = mm_transforms if mm_transforms else self.val_transforms


        if mix_video:
            video_transform = ptvt.MixVideo(num_classes=label_mapping.num_classes, mixup_alpha=0.8)
            self.train_collate = collate_function_builder(epsilon, load_frames, video_transform, position_format=position_format, relative_positions=relative_positions, team_indicator=team_indicator)
        else:
            self.train_collate = collate_function_builder(epsilon, load_frames, position_format=position_format, relative_positions=relative_positions, team_indicator=team_indicator)
        
        self.val_collate = collate_function_builder(epsilon, load_frames, position_format=position_format, relative_positions=relative_positions, team_indicator=team_indicator)


    def setup(self, stage : str):
        match stage:

            case "train":

                self.data_train = MultiModalHblDataset(
                    self.meta_path_train,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.train_transforms,
                    self.label_mapping,
                    self.overlap
                )
                self.data_val = MultiModalHblDataset(
                    self.meta_path_val,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.val_transforms,
                    self.label_mapping,
                    self.overlap,
                )
            
            case "validate":

                self.data_val = MultiModalHblDataset(
                    self.meta_path_val,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.val_transforms,
                    self.label_mapping,
                    self.overlap,
                )

            case "test":

                self.data_test = MultiModalHblDataset(
                    self.meta_path_test,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.val_transforms,
                    self.label_mapping,
                    self.overlap,
                )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=4,persistent_workers=True, pin_memory=True, shuffle=True, collate_fn=self.train_collate)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4, persistent_workers=True, collate_fn=self.val_collate)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4, collate_fn=self.val_collate)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4, collate_fn=self.val_collate)

    def teardown(self, stage: str) -> None:
        # Nothing to do here (yet)
        pass


class LitResampledHblDataset(pl.LightningDataModule):


    def __init__(self,
        meta_path : str,
        idx_mapping_train : str,
        idx_mapping_val : str,
        idx_mapping_test : str,
        label_mapping : LabelDecoder,
        seq_len : int = 16,
        sampling_rate : int = 2,
        load_frames : bool = True,
        batch_size : int = 1,
        epsilon : int = 7,
        mix_video : bool = True,
        mm_transforms : t.Compose = None,
        position_format : str = "graph_per_sequence",
        relative_positions: bool = False,
        team_indicator : bool = True,
        ) -> None:
        super().__init__()

        self.meta_path = meta_path
        self.idx_mapping_train = idx_mapping_train
        self.idx_mapping_val = idx_mapping_val
        self.idx_mapping_test = idx_mapping_test

        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.load_frames = load_frames
        self.label_mapping = label_mapping
        self.batch_size = batch_size
        self.val_transforms = t.Compose([
            mmt.FrameSequenceToTensor(),
            mmt.Resize(size=(224,224))
            ])

        self.train_transforms = mm_transforms if mm_transforms else self.val_transforms           

        if mix_video:
            video_transform = ptvt.MixVideo(num_classes=label_mapping.num_classes, mixup_alpha=0.8, cutmix_prob=0)
            self.train_collate = collate_function_builder(epsilon, load_frames, video_transform, position_format, relative_positions=relative_positions, team_indicator=team_indicator)
        else:
            self.train_collate = collate_function_builder(epsilon, load_frames, position_format=position_format, relative_positions=relative_positions, team_indicator=team_indicator)
        
        self.val_collate = collate_function_builder(epsilon, load_frames, position_format=position_format, relative_positions=relative_positions, team_indicator=team_indicator)


    def setup(self, stage : str):
        match stage:

            case "train":
                
                self.data_train = ResampledHblDataset(
                    meta_path=prepare_meta_path(self.meta_path, "train"),
                    idx_to_frame=self.idx_mapping_train,
                    label_mapping=self.label_mapping,
                    load_frames=self.load_frames,
                    seq_len=self.seq_len,
                    sampling_rate=self.sampling_rate,
                    transforms=self.train_transforms
                )
                self.data_val = ResampledHblDataset(
                    meta_path=prepare_meta_path(self.meta_path, "valid"),
                    idx_to_frame=self.idx_mapping_val,
                    label_mapping=self.label_mapping,
                    load_frames=self.load_frames,
                    seq_len=self.seq_len,
                    sampling_rate=self.sampling_rate,
                    transforms=self.val_transforms
                )
            
            case "validate":

                self.data_val = ResampledHblDataset(
                    meta_path=prepare_meta_path(self.meta_path, "valid"),
                    idx_to_frame=self.idx_mapping_val,
                    label_mapping=self.label_mapping,
                    load_frames=self.load_frames,
                    seq_len=self.seq_len,
                    sampling_rate=self.sampling_rate,
                    transforms=self.val_transforms
                )
            case "test":

                self.data_test = ResampledHblDataset(
                    meta_path=prepare_meta_path(self.meta_path, "test"),
                    idx_to_frame=self.idx_mapping_test,
                    label_mapping=self.label_mapping,
                    load_frames=self.load_frames,
                    seq_len=self.seq_len,
                    sampling_rate=self.sampling_rate,
                    transforms=self.val_transforms
                )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True, collate_fn=self.train_collate,)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4, shuffle=False, collate_fn=self.val_collate)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4, shuffle=False, collate_fn=self.val_collate)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=1, shuffle=False, collate_fn=self.val_collate)

    def teardown(self, stage: str) -> None:
        # Nothing to do here (yet)
        pass

def collate_function_builder(
        epsilon : int,
        load_frames : bool,
        mix_video : callable = None,
        position_format : str = "graph_per_sequence",
        relative_positions: bool = False,
        team_indicator:bool = True
    ):

    def multimodal_collate(instances : list):
        """Collate function that batches both position data and video data.

        Args:
            instances (list): List of instances to be batched.

        Returns:
            dict: Batched instances.
        """
        batch = {}
        for k in instances[0].keys():
            if k == "frames" and not load_frames:
                continue

            first_entry = instances[0][k]
            if isinstance(first_entry, list): # might be empty (no frames), windows_indices
                batch[k] = torch.stack([torch.Tensor(instance[k]) for instance in instances])

            elif isinstance(first_entry, torch.Tensor):
                batch[k] = torch.stack([instance[k] for instance in instances])

            elif isinstance(first_entry, np.ndarray): # frames, positions
                batch[k] = torch.stack([torch.tensor(instance[k]) for instance in instances])
                
            elif isinstance(first_entry, (int, float, np.int64, np.float64)):
                # frame_idx, query_idx, label_offset, label, match_number
                batch[k] = torch.tensor([instance[k] for instance in instances])

            elif isinstance(first_entry, PositionContainer):
                if position_format == "graph_per_sequence":
                    batch[k] = dgl.batch([instance[k].as_graph_per_sequence(epsilon, relative_positions) for instance in instances])
                if position_format == "flattened":
                    batch[k] = torch.stack([instance[k].as_flattened(normalize=True, relative_positions=relative_positions, team_indicator=team_indicator) for instance in instances])
                if position_format == "graph_per_timestep":
                    batch[k] = dgl.batch(instances[0][k].as_graph_per_timestep(epsilon))


        r = torch.rand(1).item()
        if (not mix_video is None) and r < 0.5:
            batch["frames"], batch["label"] = mix_video(batch["frames"], batch["label"])
        return batch

    return multimodal_collate


def prepare_meta_path(path : str, split : str):
    if path.endswith(".csv"):
        return path
    else:
        return Path(path) / f"meta30_{split}.csv"