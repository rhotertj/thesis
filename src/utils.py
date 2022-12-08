import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def draw_trajectory(positions : np.ndarray):
    positions[:, -1, 2] = 2 # give ball an extra team
    T = positions.shape[0]
    # start new figure
    plt.close()
    fig = plt.figure()
    plt.xlim(0,40)
    plt.ylim(0,20)

    for t in range(T):
        a = t / T
        y_t = positions[t, :, 1]
        x_t = positions[t, :, 0]
        teams = positions[t, :, 2]
        players = list(range(positions.shape[1]))
        # change axis to improve format in plot
        sns.scatterplot(y=y_t, x=x_t, hue=teams, style=players, legend=False, palette="Set1", alpha=a)

    return fig 


def array2gif(arr : np.ndarray, out_path : str, fps : int):
    # Expect tensor to be [C, F, H, W]
    if arr.ndim == 5:
        arr = arr.squeeze(0)

    duration = (1 / fps) * 1000
    
    arr = arr.swapaxes(0,1).astype(np.uint8)
    frames_arr = list(arr)

    frames = [Image.fromarray(fr.transpose(1,2,0), mode="RGB") for fr in frames_arr]
    first_frame = frames[0]
    # Each frame gets duration ms, gif is looped forever (0) (1 is loop once etc)
    first_frame.save(out_path, format="GIF", append_images=frames, save_all=True, duration=duration, loop=0, interlace=False, includ_color_table=True)
    print("Saved to", out_path)
