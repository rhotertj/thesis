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

def main(conf):
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

    lit_data = LitHandballSynced(
        **conf.data
    )
    lit_data.setup(conf.stage)

    match conf.stage:

        case "train":

            trainer.fit(
                model,
                train_dataloaders=lit_data.train_dataloader(),
                val_dataloaders=lit_data.val_dataloader()
                )
        
        case "validate":

            trainer.validate(
                model=model,
                dataloaders=lit_data.val_dataloader(),
                ckpt_path=conf.checkpoint
            )
        
        case "test":

            trainer.test(
                model=model,
                dataloaders=lit_data.test_dataloader(),
                ckpt_path=conf.checkpoint
            )


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-f", "--file", type=str, help='File that contains configuration of models, task and data.')
    parser.add_argument('-c', "--config", type=str, nargs="*", help='Overwrite config parameters of given file', required=False)

    args = parser.parse_args()
    conf = omcon.load(args.file)
    if args.config:
        cli_conf = omcon.from_dotlist(args.config)
        conf = omcon.merge(conf, cli_conf)
    main(conf)
