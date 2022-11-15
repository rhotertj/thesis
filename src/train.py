
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from position_models import Baller2Vec
from data import BuliTVPositions
from torch.utils.data import DataLoader
#https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

class LitBaller2Vec(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.model = Baller2Vec(
                [128,512],
                8,
                22,
                7140,
                7140,
                8,
                2048,
                8,
                0.5
            )

    def training_step(self, batch, batch_idx):
        wandb_logger = self.logger.experiment
        # training_step defines the train loop.
        x, y_ball, y_players = batch
        pred_ball, pred_players = self.model(x.float())
        
        loss = None
        for i in range(y_players.shape[1]):
            player_one_hot = F.one_hot(y_players[:, i], num_classes=7140).float().squeeze(0)
            new_loss = F.cross_entropy(pred_players[i], player_one_hot)
            self.log("train/part_loss", new_loss)
            if i == 0:
                loss = new_loss
            else:
                loss += new_loss
        
        ball_one_hot = F.one_hot(y_ball, num_classes=7140).float().squeeze(0)
        new_loss = F.cross_entropy(pred_ball, ball_one_hot)
        loss += new_loss
        loss = torch.mean(loss)
        self.log("train/loss", loss)

        return torch.mean(loss)

    def on_train_epoch_end(self) -> None:
        # use this to log results, like prediciton for a few examples
        return super().on_train_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# think about accumulating gradients, especially for video
# Dataset and Dataloader

if "__main__" == __name__:
    wandb_logger = WandbLogger(project="experimental", log_model="all")
    trainer = pl.Trainer(accelerator="gpu", accumulate_grad_batches=4, max_epochs=2000, logger=wandb_logger)
    dataset = BuliTVPositions()
    # TODO: Fit for batch sizes
    dataloader = DataLoader(dataset, batch_size=1)
    model = LitBaller2Vec()
    wandb_logger.watch(model)
    trainer.fit(model, dataloader)