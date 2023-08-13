# Dataset and corresponding files

The data pre-processing pipeline:

All raw data resides in `/nfs/data/mm4spa/mm_hbl/hbl_19-20`.

Data is cleaned and formatted according to availability and FPS by `preprocess_dataset.py`.
    -> output is a meta.csv full and for each split that contains all paths to positions and videos.

-> `MultiModalHblDataset` takes meta.csv

`resample_data.py` iterates over `MultiModalHblDataset` and creates a DataFrame that is saved as a jsonl file.
This dataframe contains idx, corresponding frame number and annotated event for that frame.
It can be used to resample the dataset w.r.t class imbalances and save it as another DataFrame (jsonl format).

This resampled/balanced Dataset can be used with the `ResampledHblDataset` class, which takes the jsonl file as input.

## Script usage

You can create frames from all videos by setting the desired target path in the source code and then running:

```bash
./scripts/create_frames.sh
```

Afterwards, you can create the `meta.csv` files for thw whole dataset and all splits. Make sure the desired target paths
are correct.

```bash
python scripts/preprocess_dataset.py
```

If you want to use a balanced dataset (which you should!), use the `resample_data.py` script. 

```bash
python scripts/resample_data.py --help
usage: resample_data.py [-h] [-d DATA_PATH] [--mode {matches,random,time}]
                        [--val_size VAL_SIZE] [--test_size TEST_SIZE]
                        [--background_size BACKGROUND_SIZE] [--balanced]
                        [--overlap] [--sequence_length SEQUENCE_LENGTH]
                        [--sampling_rate SAMPLING_RATE] [--upsample]

Resample the dataset!

options:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        Path for meta data.
  --mode {matches,random,time}
                        Splitting criterion.
  --val_size VAL_SIZE   Split size of validation set. Should be between 0 and
                        1, default is 0.15. Ignored when split is `matches`.
  --test_size TEST_SIZE
                        Split size of test set. Should be between 0 and 1,
                        default is 0.15. Ignored when split is `matches`.
  --background_size BACKGROUND_SIZE
                        Proportion of the background class when balancing
                        proportions. Default is 0.4.
  --balanced            Whether balancing of underrepresented classes takes
                        place.
  --overlap             Whether to use overlapping sliding windows.
  --sequence_length SEQUENCE_LENGTH
                        Sequence length.
  --sampling_rate SAMPLING_RATE
                        Sampling rate.
  --upsample            Whether to upsample shots or downsample passes.
```

To generate a dataset split by matches, with 16x2 sliding windows and balanced by upsampling shots, with 40% background instances:

```bash
python scripts/resample_data.py --mode matches --sequence_length 16 --sampling_rate 2 --balanced --background_size 0.4 --upsample
```