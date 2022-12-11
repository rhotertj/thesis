import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def draw_trajectory(positions : np.ndarray):
    """Plots player and ball positions as a scatterplot.
    Positions of the past become transparent.

    Args:
        positions (np.ndarray): Positions of shape [T, N, 3].

    Returns:
        plt.figure.Figure: The plot.
    """
    T = positions.shape[0]
    # start new figure, toss the old one
    plt.close()
    fig = plt.figure()
    # plot config
    plt.xlim(0,40)
    plt.ylim(0,20)

    positions[:, -1, 2] = 2 # give ball an extra team

    for t in range(T):
        a = t / T # transparency
        y_t = positions[t, :, 1]
        x_t = positions[t, :, 0]
        team_indicator = positions[t, :, 2] # team indicator, used for colors

        sns.scatterplot(y=y_t, x=x_t, hue=team_indicator, legend=False, palette="Set1", alpha=a)

    return fig 


def array2gif(arr : np.ndarray, out_path : str, fps : int):
    """Turns a stack of frames into a gif.

    Args:
        arr (np.ndarray): Stacked RGB frames of shape [1, C, F, H, W]
        out_path (str): Destination path for the gif.
        fps (int): Frames per second.
    """
    # Expect tensor to be [C, F, H, W]
    if arr.ndim == 5:
        arr = arr.squeeze(0)

    duration = (1 / fps) * 1000
    
    arr = arr.swapaxes(0,1).astype(np.uint8)
    frames_arr = list(arr)

    frames = [Image.fromarray(fr.transpose(1,2,0), mode="RGB") for fr in frames_arr]
    first_frame = frames[0]
    # looped forever (0) (1 is loop once etc)
    first_frame.save(out_path, format="GIF", append_images=frames, save_all=True, duration=duration, loop=0, interlace=False, includ_color_table=True)
    