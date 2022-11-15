import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

LABEL_BASE_PATH = Path("/nfs/data/mm4spa/lab-roi_estimation")
LABEL_FILE = Path("bl1415_HD_TV_annotated_0002YA_H2_foof-True_fpm-True.jsonl")


class BuliTVPositions(Dataset):

    def __init__(self, seq_len=8):
        super(BuliTVPositions).__init__()

        self.field_width = 68
        self.field_length = 105

        self.seq_len = seq_len

        print("Read", LABEL_BASE_PATH / LABEL_FILE)
        df = pd.read_json(LABEL_BASE_PATH / LABEL_FILE, lines=True, orient="records")
        df = df.iloc[:2500]
        df["halftime"] = "0002UK"
        
        self.width_linspace = np.linspace(0, self.field_width, num=self.field_width)
        self.length_linspace = np.linspace(0, self.field_length, num=self.field_length)

        self.df = df

    def __len__(self):
        return self.df.shape[0] - (self.seq_len + 1)

    def __getitem__(self, idx):
        positions = self.df["gt_pos"].iloc[idx : (idx + self.seq_len + 1)].tolist()
        # T x 23 x 2
        positions = np.array(positions)

        label_y = np.clip(positions[-1, :, 0], 0, self.field_length)
        label_x = np.clip(positions[-1, :, 1], 0, self.field_width)

        # remove label
        positions = positions[:-1]
        
        ball_positions = positions[:, :1].reshape(-1, 2)
        # add z-dimension
        z_dim = np.zeros((len(ball_positions), 1))
        ball_positions = np.hstack([ball_positions, z_dim])

        # add team indicator
        player_positions = positions[:, 1:].reshape(-1, 2)
        team_indicators = np.zeros(22)
        team_indicators[:11] = 1
        team_indicators = np.tile(team_indicators, self.seq_len).reshape(-1, 1)
        player_positions = np.hstack([player_positions, team_indicators])

        # organize ball and player positions [T * player_pos, T * ball_pos]
        positions = np.concatenate([player_positions, ball_positions])

        # bin labels
        bins_x = np.digitize(label_x, self.width_linspace)
        bins_y = np.digitize(label_y, self.length_linspace)

        # NOTE: labels depend on number of bins and not field width and length
        labels = np.clip(bins_x - 1, 0, self.field_width) * self.field_length + bins_y

        ball_label = labels[0]
        players_label = labels[1:]

        return positions, ball_label, players_label

    def export_json(self, idx):
        """Generate trajectory in json format for internal visualization tool.
        """
        positions, *_ = self.__getitem__(idx)
        # list per frame, inside coordinate lists
        # positions is T positions, T balls
        jd = {}
        team_a = [positions[f*22 : f*22 + 11, :2].tolist() for f in range(self.seq_len)]
        team_b = [positions[f*22 + 11 : (f+1) * 22, :2].tolist() for f in range(self.seq_len)]
        
        jd["team_a"] = team_a
        jd["team_b"] = team_b
        jd["balls"] = positions[-self.seq_len:].tolist()
        return jd

if __name__ == "__main__":
    import json
    data = BuliTVPositions(200)
    js = data.export_json(1)
    with open("sample.json", "w+") as f:
        json.dump(js, f)