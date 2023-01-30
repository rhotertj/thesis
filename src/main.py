import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
from pathlib import Path
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
    

    conf.model.max_epochs = conf.trainer.max_epochs

    model = LitMViT(
        **conf.model
    )

    lit_data = LitHandballSynced(
        **conf.data
    )
    lit_data.setup(conf.stage)

    callbacks = []

    # create experiment directory
    exp_dir = Path(conf.callbacks.checkpointing.dir)
    previous_experiments = [f for f in os.listdir(exp_dir) if os.path.isdir(exp_dir / f)]
    next_dir = str(len(previous_experiments) + 1)
    exp_dir = exp_dir / next_dir
    os.makedirs(exp_dir)

    if conf.save_config:
        with open(exp_dir/"config.yaml", "w+") as f:
            f.write(omcon.to_yaml(conf))


    checkpoint_cb = ModelCheckpoint(
        dirpath=exp_dir,
        every_n_epochs=conf.callbacks.checkpointing.every_n,

    )

    callbacks.append(checkpoint_cb)

    trainer = pl.Trainer(
        logger=logger,
        **conf.trainer,
        callbacks=callbacks
    )

    match conf.stage:

        case "train":

            trainer.fit(
                model,
                train_dataloaders=lit_data.train_dataloader(),
                val_dataloaders=lit_data.val_dataloader(),
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
    # overwrite file config from cli
    if args.config:
        cli_conf = omcon.from_dotlist(args.config)
        conf = omcon.merge(conf, cli_conf)
    main(conf)
