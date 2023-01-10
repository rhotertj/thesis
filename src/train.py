import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import wandb
import torch
import numpy as np

from lit_models import LitMViT
from lit_data import LitHandballSynced

def cli_main():
    cli = LightningCLI(LitMViT, LitHandballSynced, save_config_callback=None)

if "__main__" == __name__:

    cli_main()