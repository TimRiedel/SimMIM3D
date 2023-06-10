import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.upsample import SubpixelUpSample

from src.SimMIM3D.networks.masked_vit import MaskedViT3D

class SimMIM3D(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        encoder_stride: int,

    ) -> None :
        super().__init__()

        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = SubpixelUpSample(
            spatial_dims=3,
            in_channels=self.encoder.hidden_size,
            out_channels=self.encoder.in_channels,
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
        z = self.reshape_3d(z)
        x_rec = self.decoder(z)

        patch_size = self.encoder.patch_size
        mask = mask.repeat_interleave(patch_size[0], 1).repeat_interleave(patch_size[1], 2).repeat_interleave(patch_size[2], 3).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.encoder.in_channels
        return loss, x_rec


def build_simmim(cfg):
    encoder = MaskedViT3D(
        img_size=(cfg.DATA.IMG_SIZE,) * 3,
        in_channels=cfg.MODEL.IN_CHANNELS,
        patch_size=(cfg.MODEL.PATCH_SIZE,) * 3,
        dropout_rate=cfg.MODEL.ENCODER_DROPOUT,
    )

    
    return SimMIM3D(
        encoder=encoder,
        encoder_stride=cfg.MODEL.PATCH_SIZE, 
    )