import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.upsample import SubpixelUpSample


from src.mlutils.layers import PixelShuffle3D
from src.models.SimMIM.masked_vit import MaskedViT3D

class SimMIM3D(nn.Module):
    def __init__(
        self, 
        img_size: int = 96,
        in_channels: int = 4, 
        patch_size: int = 16, 
        embed_dim: int = 768, 
        dropout_rate: float = 0.0,
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
            img_size=(self.img_size,) * 3, 
            in_channels=self.in_channels, 
            patch_size=(self.patch_size,) * 3,
            embed_dim=self.embed_dim,
            dropout_rate=dropout_rate,
        )

        self.decoder = SubpixelUpSample(
            spatial_dims=3,
            in_channels=self.embed_dim,
            out_channels=4,
            scale_factor=self.encoder_stride,
            conv_block="default"
        )

    def reshape_3d(self, x):
        B, T, E = x.shape
        H = W = D = round(T ** (1./3.))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, E, H, W, D)
        return x

    
    def forward(self, x, mask):
        z = self.encoder(x, mask)

        if self.is_pretrain:
            z = self.reshape_3d(z)
            x_rec = self.decoder(z)

            mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).repeat_interleave(self.patch_size, 3).unsqueeze(1).contiguous()
            loss_recon = F.l1_loss(x, x_rec, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_channels
            return loss, x_rec

        return z