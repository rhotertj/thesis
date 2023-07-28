# MultiModal Representation Learning from Video and Position Data in Team Sports

-- Readme coming soon--
## Structure

 * `dataset` contains preprocessing scripts for the dataset.
    More on the dataset can be found [here](dataset/README.md).
 * `experiments` contains checkpoints.
 * `models` contains pre-trained models.
 * `scripts` contains scripts for setting up the code.
 * `src` contains all source code.

## Installation

Setup with conda:

```bash
$ conda env create && conda activate thesis
```
Beware, installation might take a few minutes.

Downloading models

```
$ ./scripts/download_models.sh
```

## Training

To run training, validation or testing of a model with a certain configuration, run 

```
$ python src/main.py -f config/[CONFIG].yaml -c logger.name=overwrittenName
```

Use the `-c` argument to overwrite parameters on the command line.
