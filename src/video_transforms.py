import torch
from torch import Tensor
import torchvision
import numpy

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

    def __call__(self, video):
        """
        Args:
            video (numpy.ndarray): Video to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return video_to_tensor(video)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


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

    