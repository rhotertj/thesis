# Multimodal Representation Learning from Video and Position Data in Team Sports

This repository contains the code of my master thesis. We train a variety of models to solve an action recognition
downstream task via representation learning.
In particular, we recognize shots and passes in video broadcast material of the German Handball Bundesliga and
the respective position data for players and ball.

We use a Multiscale Vision Transformer for the video data and experiment with a vanilla Transformer and multiple
Graph Neural Networks for the position data.

<p align="center">
  <img src="img/methods.svg" width="700" title="Method Overview">
</p>

## Installation

Setup with conda:

```bash
$ conda env create && conda activate thesis
```
Beware, installation might take a few (more than 15) minutes.
Consider passing `-f exact_environment.yml` in case newer versions of dependencies do not work.

While you wait, you can download the pretrained models:

```
$ ./scripts/download_models.sh
```

Have a look at [`DATASET.md`](DATASET.md) for more information regarding the data.

## Experiments

To run training, validation or testing of a model with a certain configuration, run 

```
$ python src/main.py -f config/[CONFIG].yaml -c logger.name=overwrittenValue
```

Use the `-c` argument to overwrite parameters on the command line.

You can reproduce results presented in the thesis by simply running the configurations in `config/experiments`.
Make sure to change the paths according to your data locations.

## Structure

 * `config` contains model, dataset and training configurations.
 * `notebooks` contains jupyter notebooks to claculate metrics and visualize data and model predictions.
 * `experiments` contains checkpoints and LitModel Cache from validation and test epochs.
 * `models` contains pre-trained models.
 * `scripts` contains scripts for preprocessing the dataset and downloading model checkpoints.
 * `src` contains all source code.
 * `img` contains images.