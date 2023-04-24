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
