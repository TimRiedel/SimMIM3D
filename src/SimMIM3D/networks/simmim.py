import torch.nn as nn
from monai.networks.blocks.upsample import SubpixelUpSample

from src.SimMIM3D.networks.masked_vit import MaskedViT3D

class SimMIM3D(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        decoder: nn.Module,

    ) -> None :
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reshape_embedding(self, x):
        B, T, E = x.shape
        H = W = D = round(T ** (1./3.))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, E, H, W, D)
        return x
    
    def forward(self, x, mask):
        z = self.encoder(x, mask)
        z = self.reshape_embedding(z)
        x_rec = self.decoder(z)

        return x_rec


def build_simmim(cfg):
    encoder = MaskedViT3D(
        img_size=(cfg.DATA.IMG_SIZE,) * 3,
        in_channels=cfg.MODEL.IN_CHANNELS,
        patch_size=(cfg.MODEL.PATCH_SIZE,) * 3,
        dropout_rate=cfg.MODEL.ENCODER_DROPOUT,
    )

    decoder = SubpixelUpSample(
        spatial_dims=3,
        in_channels=encoder.hidden_size,
        out_channels=encoder.in_channels,
        scale_factor=cfg.MODEL.PATCH_SIZE,
        conv_block="default"
    )
    
    return SimMIM3D(
        encoder=encoder,
        decoder=decoder
    )