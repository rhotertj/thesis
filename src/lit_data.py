from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms as t
from video_transforms import FrameSequenceToTensor, NormalizeVideo

from data import MultiModalHblDataset, LabelDecoder

class LitHandballSynced(pl.LightningDataModule):

    def __init__(
        self,
        meta_path_train : str,
        meta_path_val : str,
        meta_path_test : str,
        label_mapping : LabelDecoder,
        seq_len : int = 8,
        sampling_rate : int = 1,
        load_frames : bool = True,
        batch_size : int = 1
        ) -> None:

        super().__init__()
        self.meta_path_train = meta_path_train
        self.meta_path_val = meta_path_val
        self.meta_path_test = meta_path_test

        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.load_frames = load_frames
        self.label_mapping = label_mapping
        self.batch_size = batch_size
        self.transforms = t.Compose([
            FrameSequenceToTensor(),
            t.Resize((224,224)),
            # t.CenterCrop(224),
            # NormalizeVideo(
            #     mean=[0.39449842, 0.4566527, 0.49926605],
            #     std=[0.18728599, 0.21862774, 0.267905]
            #     ),
            ])


    def setup(self, stage : str):
        match stage:

            case "train":

                self.data_train = MultiModalHblDataset(
                    self.meta_path_train,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.transforms,
                    self.label_mapping
                )
                self.data_val = MultiModalHblDataset(
                    self.meta_path_val,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.transforms,
                    self.label_mapping
                )
            
            case "validate":

                self.data_val = MultiModalHblDataset(
                    self.meta_path_val,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.transforms,
                    self.label_mapping
                )

            case "test":

                self.data_test = MultiModalHblDataset(
                    self.meta_path_test,
                    self.seq_len,
                    self.sampling_rate,
                    self.load_frames,
                    self.transforms,
                    self.label_mapping
                )

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

