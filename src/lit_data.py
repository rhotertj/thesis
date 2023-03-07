from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pytorch_lightning as pl
import torch
import numpy as np
import dgl
from torchvision import transforms as t
import video_transforms as vt
import pytorchvideo.transforms as ptvt

from data.datasets import MultiModalHblDataset, ResampledHblDataset
from data.labels import LabelDecoder
from data.data_utils import create_graph

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
        mix_video : bool = True
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
        self.transforms = t.Compose([
            vt.FrameSequenceToTensor(),
            t.Resize((224,224)),
            vt.TimeFirst(),
            t.ColorJitter(brightness=0.3, hue=.3, contrast=0.3, saturation=0.3),
            ptvt.RandAugment(),
            vt.ChannelFirst()
            # NormalizeVideo(
            #     mean=[0.39449842, 0.4566527, 0.49926605],
            #     std=[0.18728599, 0.21862774, 0.267905]
            #     ),
            ])

        if mix_video:
            video_transform = ptvt.MixVideo(num_classes=label_mapping.num_classes)
            self.collate_func = collate_function_builder(epsilon, load_frames, video_transform)

        self.collate_func = collate_function_builder(epsilon, load_frames)


    def setup(self, stage : str):
        match stage:

            case "train":

                self.data_train = MultiModalHblDataset(
                    self.meta_path_train,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.transforms,
                    self.label_mapping,
                    self.overlap
                )
                self.data_val = MultiModalHblDataset(
                    self.meta_path_val,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.transforms,
                    self.label_mapping,
                    self.overlap,
                )
            
            case "validate":

                self.data_val = MultiModalHblDataset(
                    self.meta_path_val,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.transforms,
                    self.label_mapping,
                    self.overlap,
                )

            case "test":

                self.data_test = MultiModalHblDataset(
                    self.meta_path_test,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.transforms,
                    self.label_mapping,
                    self.overlap,
                )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=4,persistent_workers=True, pin_memory=True, shuffle=True, collate_fn=self.collate_func)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4, persistent_workers=True, collate_fn=self.collate_func)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_func)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_func)

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
        mix_video : bool = True
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
        self.transforms = t.Compose([
            vt.FrameSequenceToTensor(),
            t.Resize((224,224)),
            vt.TimeFirst(),
            t.ColorJitter(brightness=0.3, hue=.3, contrast=0.3, saturation=0.3),
            ptvt.RandAugment(),
            vt.ChannelFirst()
            # NormalizeVideo(
            #     mean=[0.39449842, 0.4566527, 0.49926605],
            #     std=[0.18728599, 0.21862774, 0.267905]
            #     ),
            ])

        if mix_video:
            video_transform = ptvt.MixVideo(num_classes=label_mapping.num_classes)
            self.collate_func = collate_function_builder(epsilon, load_frames, video_transform)
            
        self.collate_func = collate_function_builder(epsilon, load_frames)

    def setup(self, stage : str):
        match stage:

            case "train":

                self.data_train = ResampledHblDataset(
                    meta_path=Path(self.meta_path) / "meta3d_train.csv",
                    idx_to_frame=self.idx_mapping_train,
                    label_mapping=self.label_mapping,
                    load_frames=self.load_frames,
                    seq_len=self.seq_len,
                    sampling_rate=self.sampling_rate,
                    transforms=self.transforms
                )
                self.data_val = ResampledHblDataset(
                    meta_path=Path(self.meta_path) / "meta3d_val.csv",
                    idx_to_frame=self.idx_mapping_val,
                    label_mapping=self.label_mapping,
                    load_frames=self.load_frames,
                    seq_len=self.seq_len,
                    sampling_rate=self.sampling_rate,
                    transforms=self.transforms
                )
            
            case "validate":

                self.data_val = ResampledHblDataset(
                    meta_path=Path(self.meta_path) / "meta3d_val.csv",
                    idx_to_frame=self.idx_mapping_val,
                    label_mapping=self.label_mapping,
                    load_frames=self.load_frames,
                    seq_len=self.seq_len,
                    sampling_rate=self.sampling_rate,
                    transforms=self.transforms
                )
            case "test":

                self.data_test = ResampledHblDataset(
                    meta_path=Path(self.meta_path) / "meta3d_test.csv",
                    idx_to_frame=self.idx_mapping_test,
                    label_mapping=self.label_mapping,
                    load_frames=self.load_frames,
                    seq_len=self.seq_len,
                    sampling_rate=self.sampling_rate,
                    transforms=self.transforms
                )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=4,persistent_workers=True, pin_memory=True, shuffle=True, collate_fn=self.collate_func,)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4, persistent_workers=True, collate_fn=self.collate_func)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_func)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_func)

    def teardown(self, stage: str) -> None:
        # Nothing to do here (yet)
        pass

def collate_function_builder(epsilon : int, load_frames : bool, mix_video : callable = None):

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
                if k == "positions":
                    graphs = [create_graph(torch.Tensor(instance[k]), epsilon) for instance in instances]
                    batch[k] = dgl.batch(graphs)
                else:
                    batch[k] = torch.stack([torch.tensor(instance[k]) for instance in instances])
                
            elif isinstance(first_entry, (int, float, np.int64, np.float64)):
                # frame_idx, query_idx, label_offset, label, match_number
                batch[k] = torch.tensor([instance[k] for instance in instances])

        if mix_video:
            batch["frames"], batch["label"] = mix_video(batch["frames"], batch["label"])
        return batch

    return multimodal_collate