import torch
import torch.nn as nn
from typing import Sequence, Union

from src.models.SimMIM.masked_vit import MaskedViT3D

class SimMIM3D(nn.Module):
    def __init__(
        self, 
        img_size: Union[Sequence[int], int] = (96, 96, 96),
        in_channels: int = 4, 
        patch_size: int = 16, 
        embed_dim: int = 768, 
        is_pretrain: bool = True
    ) -> None :
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels 
        self.patch_size = patch_size 
        self.embed_dim = embed_dim
        self.encoder_stride = patch_size
        self.is_pretrain = is_pretrain

        self.encoder = MaskedViT3D(
            img_size=self.img_size, 
            in_channels=self.in_channels, 
            patch_size=(self.patch_size,) * 3,
            embed_dim=self.embed_dim,
        )

        self.decoder = nn.ConvTranspose3d(
            in_channels=self.embed_dim,
            out_channels=self.in_channels,
            kernel_size=self.patch_size,
            stride=self.encoder_stride,
        )

    def reshape_3d(self, x):
        B, T, E = x.shape
        H = W = D = round(T ** (1./3.))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, E, H, W, D)
        return x

    
    def forward(self, x, mask):
        z = self.encoder(x, mask)
        z = self.reshape_3d(z)

        if self.is_pretrain:
            x_rec = self.decoder(z)
            return x_rec

        return z