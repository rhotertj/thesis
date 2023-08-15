# Multimodal Representation Learning from Video and Position Data in Team Sports

#### Abstract
*Data-driven methods have become increasingly important for sports analysis, including the*
*assessment of player and team performance. So far, related work has focused on solving tasks*
*solely on a single data domain, i.e., player position or video data, whereas the strengths of*
*combining multiple modalities remain mostly unexplored. Moreover, approaches using position*
*data have task-specific architectures and rely on handcrafted features. This thesis aims*
*to learn latent representations for video and position data, which can be utilized to solve other*
*downstream tasks without training from scratch or modifying the networks architecture. Since*
*actions like shots and passes fundamentally characterize a match, we use action recognition*
*as a pretraining task. We reproduce state-of-the-art results from the SoccerNet action spotting*
*challenge for our video data and explore the ability of a Transformer and Graph Neural*
*Networks for learning a representation of raw position data. Finally, we present a multimodal*
*variant that significantly outperforms the unimodal approaches in recognizing actions*
*as well as the downstream task of action spotting. Experiments are performed on unedited*
*broadcast video material and corresponding synchronized position data of 25 halves from the*
*German Handball Bundesliga.*

## Method Overview

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