import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import wandb
import torch
from torchvision.transforms.functional import center_crop
import numpy as np
import seaborn as sns

from video_models import make_kinetics_mvit


class LitMViT(pl.LightningModule):

    def __init__(
        self,
        pretrained_path : str,
        learning_rate : float,
        num_classes : int
    ) -> None:
        super().__init__()
        self.model = make_kinetics_mvit(pretrained_path, num_classes)

        # if self.log is not None:
        #     wandb.watch(self.model)

        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # print(batch)
        x = batch["frames"]
        print(x.shape)
        targets = batch["label"]
        print(targets)
        y = self.model(x)
        
        loss = self.loss(y, targets)
        self.log("train/batch_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log("train/epoch_loss", np.mean([item["loss"].item() for item in outputs]))

    def validation_step(self, batch, batch_idx):
        pass
        # loss = ...
        # acc = ...
        # fig = ...
        # wandb.log({"val/plot_trajectory": wandb.Image(fig)})

        # return {"loss" : loss.item(), "acc" : acc.item()}

    def validation_epoch_end(self, outputs) -> None:
        pass
        # self.log("val/acc", np.mean([output["acc"] for output in outputs]))
        # self.log("val/loss", np.mean([output["loss"] for output in outputs]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
