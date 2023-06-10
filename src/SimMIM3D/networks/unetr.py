from monai.networks.nets import UNETR

from src.SimMIM3D.networks.masked_vit import MaskedViT3D

class UNETR3D(UNETR):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        feature_size: int = 16,
        dropout_rate: float = 0.0,
        freeze_encoder: bool = True,
    ) -> None :
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(img_size,) * 3,
            feature_size=feature_size,
            dropout_rate=dropout_rate,
        )
        if freeze_encoder:
            for param in self.vit.parameters():
                param.requires_grad = False
    

def build_unetr(cfg):
    return UNETR3D(
        in_channels=cfg.MODEL.IN_CHANNELS,
        out_channels=cfg.DATA.NUM_CLASSES,
        img_size=cfg.DATA.IMG_SIZE,
        feature_size=cfg.MODEL.PATCH_SIZE,
        dropout_rate=cfg.MODEL.ENCODER_DROPOUT,
    )
