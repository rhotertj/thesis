import os
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf as omcon
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from torchvision import transforms as t
import video_transforms as vt
import pytorchvideo.transforms as ptvt

from lit_models import LitModel, weighted_cross_entropy, unweighted_cross_entropy, twin_head_loss
from video_models import make_kinetics_mvit
from graph_models import GAT, PositionTransformer, GIN, GCN
from multimodal_models import MultiModalModel
from lit_data import LitMultiModalHblDataset, LitResampledHblDataset
from data.labels import LabelDecoder
from utils import get_proportions_df


def main(conf):
    pl.seed_everything(conf.seed_everything)
    torch.set_float32_matmul_precision("high")

    logger = WandbLogger(
        **conf.logger,
        config=conf
    )
    experiment_name = logger.experiment.name
    

    label_decoder = LabelDecoder(conf.num_classes)

    model = eval(conf.model.name)(**conf.model.params, num_classes=conf.num_classes, batch_size=conf.data.params.batch_size)

    optimizer = eval(conf.optimizer.name)(**conf.optimizer.params, params=model.parameters())
    if not conf.scheduler is None:
        scheduler = eval(conf.scheduler.name)(T_max=conf.trainer.max_epochs, optimizer=optimizer, last_epoch=-1)
    else:
        scheduler = None

    loss_func = eval(conf.loss_func)

    lit_model = LitModel(
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        model=model,
        label_mapping=label_decoder
    )

    if conf.data.transforms:
        video_transforms = []
        for transform in conf.data.transforms:
            if transform.get("params", False):
                trans = eval(transform.name)(**transform.params)
            else:
                trans = eval(transform.name)()

            video_transforms.append(trans)
        video_transforms = t.Compose(video_transforms)
    else:
        video_transforms = None
    print("Transforms:", video_transforms)

    lit_data = eval(conf.data.name)(
        **conf.data.params,
        video_transforms=video_transforms,
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
    exp_dir = Path([cb.params.dirpath for cb in conf.callbacks if cb.name == "ModelCheckpoint"][0])
    # previous_experiments = [f for f in os.listdir(exp_dir) if os.path.isdir(exp_dir / f)]
    # next_dir = str(len(previous_experiments) + 1)
    exp_dir = exp_dir / experiment_name
    os.makedirs(exp_dir, exist_ok=True) # allow overwrite, useful for manually set expnames

    if conf.save_config:
        with open(exp_dir/"config.yaml", "w+") as f:
            f.write(omcon.to_yaml(conf))

    for callback in conf.callbacks:
        cb = eval(callback.name)(**callback.params)
        callbacks.append(cb)

    trainer = pl.Trainer(
        logger=logger,
        **conf.trainer,
        callbacks=callbacks
    )

    match conf.stage:

        case "train":

            trainer.fit(
                model=lit_model,
                train_dataloaders=lit_data.train_dataloader(),
                val_dataloaders=lit_data.val_dataloader(),
                )
        
        case "validate":

            trainer.validate(
                model=lit_model,
                dataloaders=lit_data.val_dataloader(),
                ckpt_path=conf.checkpoint
            )
        
        case "test":

            trainer.test(
                model=lit_model,
                dataloaders=lit_data.test_dataloader(),
                ckpt_path=conf.checkpoint
            )


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Run pytorch lightning pipeline according to configuration.')
    parser.add_argument("-f", "--file", type=str, help='File that contains configuration of models, task and data.')
    parser.add_argument('-c', "--config", type=str, nargs="*", help='Overwrite config parameters of given file', required=False)

    args = parser.parse_args()
    conf = omcon.load(args.file)
    # overwrite file config from cli
    if args.config:
        cli_conf = omcon.from_dotlist(args.config)
        conf = omcon.merge(conf, cli_conf)
    main(conf)
