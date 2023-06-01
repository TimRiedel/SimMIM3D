import torch
import torch.nn as nn
from typing import Sequence, Union
from monai.networks.nets import ViT

class MaskedViT3D(ViT):
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int] = (96, 96, 96),
        patch_size: Union[Sequence[int], int] = (16, 16, 16),
        embed_dim: int = 768,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=embed_dim,
            dropout_rate=dropout_rate,
            pos_embed=pos_embed,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


    def apply_mask(self, x, mask):
        B, T, _ = x.shape
        mask_token = self.mask_token.expand(B, T, -1)
        w = mask.unsqueeze(-1).expand(-1, -1, 768)
        x = x * (1 - w) + mask_token * w
        return x

    def forward_embedding(self, x):
        # Source: https://docs.monai.io/en/stable/_modules/monai/networks/nets/vit.html
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, mask):
        x = self.patch_embedding(x)
        x = self.apply_mask(x, mask)
        x = self.forward_embedding(x)
        return x
        
    
