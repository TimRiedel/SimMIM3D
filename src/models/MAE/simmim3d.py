import torch
import torch.nn as nn
from monai.networks.nets import ViT, ViTAutoEnc
from monai.networks.blocks import SubpixelUpsample

class SimMIM3D(nn.Module):
    def __init__(self, img_size, in_channels=4, patch_size=(16, 16, 16), should_pretrain=True):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels 
        self.patch_size = patch_size 
        self.encoder_stride = patch_size[0]
        self.should_pretrain = should_pretrain
        self.hidden_size = 768

        self.encoder = ViT(
            img_size=img_size, 
            in_channels=in_channels, 
            # TODO: higher patch size reduces computation needs
            patch_size=patch_size,
            hidden_size=self.hidden_size,
            pos_embed='conv', 
        )

        self.norm = nn.LayerNorm(self.hidden_size)

        self.decoder = nn.ConvTranspose3d(
            in_channels=self.hidden_size,
            out_channels=in_channels,
            kernel_size=self.encoder_stride,
            stride=self.encoder_stride,
        )

    def reshape_3d(self, x):
        # x shape: (B, 216, 768)
        x = x.transpose(1, 2)
        # x shape: (B, 768, 216)
        d = [s // p for s, p in zip(self.img_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        # x shape: (B, 768, 6, 6, 6)
        return x
    
    def forward(self, x):
        z, _ = self.encoder(x)
        z = self.norm(z)
        z = self.reshape_3d(z)

        if self.should_pretrain:
            x_rec = self.decoder(z)
            return z, x_rec

        return z