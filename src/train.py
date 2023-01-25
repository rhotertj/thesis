import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
import numpy as np

from lit_models import LitMViT
from lit_data import LitHandballSynced
from omegaconf import OmegaConf as omcon
import argparse


def cli_main():
    cli = LightningCLI(LitMViT, LitHandballSynced, save_config_callback=None)



def read_config(conf_path):
    conf = omcon.load(conf_path)
    return conf

def main(conf_path):

    conf = read_config(conf_path)

    pl.seed_everything(conf.seed_everything)

    

    logger = WandbLogger(
        **conf.logger
    )
    trainer = pl.Trainer(
        logger=logger,
        **conf.trainer
    )

    conf.model.max_epochs = conf.trainer.max_epochs

    model = LitMViT(
        **conf.model
    )

    if conf.checkpoint:
        model.load_from_checkpoint(conf.checkpoint)

    lit_data = LitHandballSynced(
        **conf.data
    )
    lit_data.setup(conf.stage)

    trainer.fit(
        model,
        train_dataloaders=lit_data.train_dataloader(),
        val_dataloaders=lit_data.val_dataloader()
        )

if "__main__" == __name__:
    main("config/mvit_omega.yaml")
