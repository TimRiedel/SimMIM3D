import torch
import torch.nn as nn
from torch import Tensor
from monai.networks.utils import pixelshuffle

class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return pixelshuffle(input, spatial_dims=3, scale_factor=self.upscale_factor) 