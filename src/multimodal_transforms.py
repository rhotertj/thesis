from typing import Any
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as t
import numpy
import random


"""
These classes are adapted from the torchvision image transforms.
Unlike the claims in the documentation regarding the inputs shape, 
torchvision has many sanity checks for the channel dimension to be at dimension 0.
This collides with the temporal dimension in videos.
'Irrelevant' parts or checks from the original codebase are omitted, just like jit flags.
Some classes only serve as wrappers to allow a single stack of transforms for both modalities.
"""

class ChannelFirst:
    """Transpose a video from shape [T, C, H, W] to [C, T, H, W].
    """    

    def __init__(self) -> None:
        pass

    def __call__(self, mm_data) -> torch.Tensor:
        if not "frames" in mm_data or mm_data["frames"] == []:
            return mm_data
        if not mm_data["frames"].shape[1] in (1,3):
            raise ValueError("Channel dimension expected at dimension 1.")
        mm_data["frames"] = torch.einsum("tchw->cthw", mm_data["frames"])
        return mm_data

    def __repr__(self) -> str:
        return "[T, C, H, W] -> [C, T, H, W]"


class TimeFirst:
    """Transpose a video from shape [C, T, H, W] to [T, C, H, W].
    """    
    def __init__(self) -> None:
        pass

    def __call__(self, mm_data) -> torch.Tensor:
        if not "frames" in mm_data or mm_data["frames"] == []:
            return mm_data
        if not mm_data["frames"].shape[0] in (1,3):
            raise ValueError("Channel dimension expected at dimension 0.")
        mm_data["frames"] = torch.einsum("cthw->tchw", mm_data["frames"])
        return mm_data

    def __repr__(self) -> str:
        return "[C, T, H, W] -> [T, C, H, W]"

class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, mm_data):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            if "frames" in mm_data and not mm_data["frames"] == []:
                mm_data["frames"] = F.hflip(mm_data["frames"])
                mm_data["positions"] = mm_data["positions"].mirror_again(horizontal=True, vertical=False)
        return mm_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class FrameSequenceToTensor:
    """Convert a ``numpy.ndarray`` to tensor. This transform does not support torchscript.
    Converts a numpy.ndarray (T x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0]
    if the numpy.ndarray has dtype = np.uint8.
    In the other cases, tensors are returned without scaling.
    """

    def __init__(self) -> None:
        # _log_api_usage_once(self)
        pass

    def __call__(self, mm_data):
        """
        Args:
            video (numpy.ndarray): Video to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if not "frames" in mm_data or mm_data["frames"] == []:
            return mm_data
        mm_data["frames"] = video_to_tensor(mm_data["frames"])
        return mm_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, mm_data):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        if not "frames" in mm_data or mm_data["frames"] == []:
            return mm_data
        mm_data["frames"] = normalize_video(mm_data["frames"], self.mean, self.std, self.inplace)
        return mm_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"

class Resize:

    def __init__(self, **kwargs) -> None:
        self.transform = t.Resize(**kwargs)

    def __call__(self, mm_data) -> dict:
        if not "frames" in mm_data or mm_data["frames"] == []:
            return mm_data
        mm_data["frames"] = self.transform(mm_data["frames"])
        return mm_data

    def __repr__(self) -> str:
        return f"{self.transform}"
    
class ColorJitter:

    def __init__(self, **kwargs) -> None:
        self.transform = t.ColorJitter(**kwargs)

    def __call__(self, mm_data) -> Any:
        if not "frames" in mm_data or mm_data["frames"] == []:
            return mm_data
        mm_data["frames"] = self.transform(mm_data["frames"])
        return mm_data
    
    def __repr__(self) -> str:
        return f"{self.transform}"

class Translate:

    def __init__(self, mean=0, y_std=1, x_std=2) -> None:
        self.mean = mean
        self.x_std = x_std
        self.y_std = y_std
        
    def __call__(self, mm_data) -> Any:
        translate_x = numpy.random.normal(self.mean, self.x_std)
        translate_y = numpy.random.normal(self.mean, self.y_std)

        mm_data["positions"].team_a[:, :, 0] += translate_x # longer side, x
        mm_data["positions"].team_a[:, :, 1] += translate_y

        #positions.team_a = torch.clamp(positions.team_a, min=0, max=1)
        mm_data["positions"].team_b[:, :, 0] += translate_x # longer side, x
        mm_data["positions"].team_b[:, :, 1] += translate_y

        mm_data["positions"].ball[:, :, 0] += translate_x # longer side, x
        mm_data["positions"].ball[:, :, 1] += translate_y

        return mm_data
    
    def __repr__(self) -> str:
        return "Translate"

class Shuffle:

    def __init__(self) -> None:
        pass

    def __call__(self, mm_data) -> Any:
        mm_data["positions"].player_order = torch.randperm(mm_data["positions"].N)
        return mm_data
    
    def __repr__(self) -> str:
        return "Shuffle"

def normalize_video(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip

def video_to_tensor(video):

    if not isinstance(video, numpy.ndarray):
        raise TypeError(f"Not implemented for {type(video)}")
    
    if not video.ndim == 4 and video.shape[-1] == 3:
        raise ValueError("Video should be RGB and of shape (T x H x W x C)")

    video = torch.from_numpy(video.transpose(3, 0, 1, 2)).contiguous()
    if isinstance(video, torch.ByteTensor) or isinstance(video, torch.uint8):
        return video.to(torch.float32).div(255)
    else:
        return video

    