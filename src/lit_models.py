import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import wandb
import torch
from torchvision.transforms.functional import center_crop
import numpy as np
import seaborn as sns
import PIL

from utils import draw_trajectory
from video_models import make_kinetics_mvit


class LitMViT(pl.LightningModule):

    def __init__(
        self,
        pretrained_path : str,
        learning_rate : float,
        num_classes : int,
        momentum : float,
        weight_decay : float,
        max_epochs : int
    ) -> None:
        super().__init__()
        self.model = make_kinetics_mvit(pretrained_path, num_classes)

        # if self.log is not None:
        #     wandb.watch(self.model)

        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # print(batch)
        x = batch["frames"]
        targets = batch["label"]
        y = self.model(x)
        
        loss = self.loss(y, targets)
        self.log("train/batch_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        print("train/worst10", torch.tensor(np.argsort([item["loss"].item() for item in outputs])[:-10]))
        self.log("train/epoch_loss", np.mean([item["loss"].item() for item in outputs]))

    def validation_step(self, batch, batch_idx):
        x = batch["frames"]
        targets = batch["label"]
        y = self.model(x)
        
        loss = self.loss(y, targets)
        self.log("val/batch_loss", loss)
        acc = torch.sum(targets == y.argmax(-1)) / len(targets)
        self.log("val/batch_acc", acc)

        if batch_idx == 0:
            frames = x.detach().cpu().numpy()
            frames = (frames * 255).astype(np.uint8)
            trajectory = batch["positions"].detach().cpu().numpy()

            for b in range(batch["frames"].shape[0]):
                positions = trajectory[b]
                fig = draw_trajectory(positions)
                wandb.log({"val/trajectory": wandb.Image(fig)})

                images = frames[b].transpose(1, 0, 2, 3)
                wandb.log({"video": wandb.Video(images, fps=10, format="gif", caption=f"Predicted: {y[b]}, Ground Truth: {targets[b]}, Index {batch['frame_idx']} {batch['match_number']}")})
        return {"loss" : loss.item(), "acc" : acc.item()}

    def validation_epoch_end(self, outputs) -> None:
        # TODO: Log confusion matrix
        self.log("val/acc", np.mean([output["acc"] for output in outputs]))
        self.log("val/loss", np.mean([output["loss"] for output in outputs]))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=(self.lr or self.learning_rate),
            momentum = self.momentum,
            weight_decay = self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]
