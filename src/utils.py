import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import numpy as np
from PIL import Image
import torch.utils.data
import networkx as nx
from data import PositionContainer, LabelDecoder
from floodlight.vis.pitches import plot_handball_pitch

def get_proportions_df(
    dataset: torch.utils.data.Dataset,
    label_decoder: LabelDecoder,
    num_classes: int,
) -> pd.DataFrame:
    """Return a loggable `pd.DataFrame` that contains class proportions of the given Dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        label_decoder (LabelDecoder): Label decoder that maps class integers to names.
        num_classes (int): Number of classes.

    Returns:
        pd.DataFrame: The proportions DataFrame
    """    
    cnt = dataset.get_class_proportions()

    proportions = []
    for cls, n in cnt.most_common(num_classes):
        proportions.append({"name": label_decoder.class_names[cls], "n": n, "proportion": n / len(dataset)})

    return pd.DataFrame(proportions)

def draw_trajectory(positions: PositionContainer, ax=None, colors = None):
    """Plots player and ball positions as a scatterplot.
    Positions of the past become transparent.

    Args:
        positions (PositionContainer): Positions of shape [T, N, 3].

    Returns:
        plt.figure.Figure: The plot.
    """
    team_a = positions.team_a
    team_b = positions.team_b
    ball = positions.ball
    T = team_a.shape[0]
    # start new figure, toss the old one
    if ax is None:
        ax = plt.subplots()[1]
    plot_handball_pitch(xlim=(0,40), ylim=(0,20), unit='m', color_scheme='standard', show_axis_ticks=True, ax=ax)
    # plot config
    # plt.xlim(0, 40)
    # plt.ylim(0, 20)
    if colors is None:
        colors = ["red", "green", "blue"]
    for i, pos in enumerate([team_a, team_b, ball]):
        for t in range(T):
            for agent in range(pos.shape[1]):
                a = t / T  # transparency
                y_t = pos[t, agent, 1].item()
                x_t = pos[t, agent, 0].item()
                sns.scatterplot(
                    y=[y_t],
                    x=[x_t],
                    color=colors[i],
                    legend=False,
                    alpha=a,
                    ax=ax
                )

    return ax

def array2gif(arr: np.ndarray, out_path: str, fps: int):
    """Turns a stack of frames into a gif.

    Args:
        arr (np.ndarray): Stacked RGB frames of shape [1, C, T, H, W]
        out_path (str): Destination path for the gif.
        fps (int): Frames per second.
    """
    # Expect tensor to be [C, T, H, W]
    if arr.ndim == 5:
        arr = arr.squeeze(0)

    shape = arr.shape
    # transpose to thwc
    if shape[1] == 3:  # expect T C H W
        arr = arr.transpose(0, 2, 3, 1)
    if shape[0] == 3:  # expect C T H W
        arr = arr.transpose(1, 2, 3, 0)

    duration = (1 / fps) * 1000

    arr = arr.astype(np.uint8)  # [T x H x W x C]
    frames_arr = list(arr)

    frames = [Image.fromarray(fr, mode="RGB") for fr in frames_arr]
    first_frame = frames[0]
    # looped forever (0) (1 is loop once etc)
    first_frame.save(
        out_path,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
        interlace=False,
        includ_color_table=True
    )


def graph2gif(G : nx.Graph, positions : np.ndarray, out_path: str, fps: int):
    """Turns a graph and its corresponding node positions into a gif.

    Args:
        G (nx.Graph): Graph to be plotted.
        arr (np.ndarray): Stacked RGB frames of shape [1, C, T, H, W]
        out_path (str): Destination path for the gif.
        fps (int): Frames per second.
    """
    def node_color(p):
        if p == 1:
            return "green"
        elif p == 0:
            return "red"
        return "blue"
    
    images = []
    for t in range(16):
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        nx.draw(
            G,
            with_labels=True,
            font_weight='bold',
            pos=[positions[t, n, :2] for n in range(15)],
            node_color=[node_color(positions[t, n, 2]) for n in range(15)],
        )
        
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = int(width), int(height)
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        images.append(image)
        plt.clf()

    images = np.stack(images)
    return 0 