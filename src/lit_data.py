from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms
from video_transforms import FrameSequenceToTensor, NormalizeVideo
from utils import has_action, shot_pass_background

from data import MultiModalHblDataset

class LitHandballSynced(pl.LightningDataModule):
    def __init__(self, meta_path : str, seq_len : int = 8, sampling_rate : int = 1, load_frames : bool = True, batch_size : int = 1) -> None:
        super().__init__()
        self.meta_path = meta_path
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.load_frames = load_frames
        self.batch_size = batch_size
        self.transforms = transforms.Compose([FrameSequenceToTensor(), transforms.CenterCrop(224), NormalizeVideo([0.39449842, 0.4566527, 0.49926605], [0.18728599, 0.21862774, 0.267905])]) # TODO add normalize


    def setup(self, stage : str):
        self.data_full = MultiModalHblDataset(
            self.meta_path,
            self.seq_len,
            self.sampling_rate,
            self.load_frames,
            self.transforms,
            shot_pass_background
        )
        # TODO: Choose games for train val test split and create individual meta files for them
        # Use 'stage' to load test only or train val or both
        # Think about transforms for both modalities :)
        self.data_train, self.data_val, self.data_test = random_split(self.data_full, [0.7, 0.15, 0.15])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=4,persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)

    def teardown(self, stage: str) -> None:
        # Nothing to do here (yet)
        pass

