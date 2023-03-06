import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf as omcon
import argparse

from lit_models import LitMViT, LitGAT
from lit_data import LitMultiModalHblDataset, LitResampledHblDataset
from data.labels import LabelDecoder
from utils import get_proportions_df


def main(conf):
    pl.seed_everything(conf.seed_everything)

    logger = WandbLogger(
        **conf.logger,
        config=conf
    )
    experiment_name = logger.experiment.name
    conf.model.max_epochs = conf.trainer.max_epochs

    label_decoder = LabelDecoder(conf.num_classes)

    model = eval(conf.model_class)(
        **conf.model,
        label_mapping=label_decoder
    )

    lit_data = eval(conf.dataset)(
        **conf.data,
        label_mapping=label_decoder
    )
    lit_data.setup(conf.stage)

    if conf.log_proportions:
        train_df = get_proportions_df(lit_data.data_train, label_decoder, conf.num_classes)
        val_df = get_proportions_df(lit_data.data_val, label_decoder, conf.num_classes)
        logger.log_table("train/classes", data=train_df)
        logger.log_table("val/classes", data=val_df)
    
    callbacks = []

    # create experiment directory
    exp_dir = Path(conf.callbacks.checkpointing.dir)
    # previous_experiments = [f for f in os.listdir(exp_dir) if os.path.isdir(exp_dir / f)]
    # next_dir = str(len(previous_experiments) + 1)
    exp_dir = exp_dir / experiment_name
    os.makedirs(exp_dir, exist_ok=True) # allow overwrite, useful for manually set expnames

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
