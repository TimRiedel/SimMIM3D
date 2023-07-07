from monai.networks.nets import UNETR
import torch

from src.SimMIM3D.networks.masked_vit import MaskedViT3D

class UNETR3D(UNETR):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        feature_size: int = 16,
        dropout_rate: float = 0.0,
        encoder_weights: dict | None = None,
    ) -> None :
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(img_size,) * 3,
            feature_size=feature_size,
            dropout_rate=dropout_rate,
        )
        if encoder_weights is not None:
            print("Loading ViT encoder weights...")
            incompatible_keys = self.vit.load_state_dict(encoder_weights, strict=False)
            print(f"CAUTION: Incompatible keys detected for ViT encoder: {incompatible_keys}\n")
 

def build_unetr(cfg):
    if cfg.MODEL.ENCODER_CKPT_PATH != "":
        checkpoint = torch.load(cfg.MODEL.ENCODER_CKPT_PATH)
        state_dict = checkpoint["state_dict"]

        # assert that the checkpoint has keys starting with 'net.encoder'
        if not any([k.startswith('net.encoder') for k in state_dict.keys()]):
            raise ValueError(f"Checkpoint at {cfg.MODEL.ENCODER_CKPT_PATH} does not contain a compatible ViT encoder whichs states are starting with 'net.encoder.'")
        if not any([k.startswith('net.encoder.patch_embedding') for k in state_dict.keys()]):
            raise ValueError(f"Checkpoint at {cfg.MODEL.ENCODER_CKPT_PATH} does not contain a compatible ViT encoder whichs states are starting with 'net.encoder.patch_embedding'")

        # get all items that start with 'net.encoder'
        encoder_weights = {k: v for k, v in state_dict.items() if k.startswith('net.encoder.')}

        # remove prefix 'net.encoder.' from keys
        encoder_weights = {k.replace('net.encoder.', ''): v for k, v in encoder_weights.items()}

        # remove all items that start with patch_embedding
        encoder_weights = {k: v for k, v in encoder_weights.items() if not k.startswith('patch_embedding')}
        print(encoder_weights.keys())
    else:
        encoder_weights = None

    return UNETR3D(
        in_channels=cfg.MODEL.IN_CHANNELS,
        out_channels=cfg.DATA.NUM_CLASSES,
        img_size=cfg.DATA.IMG_SIZE,
        feature_size=cfg.MODEL.PATCH_SIZE,
        dropout_rate=cfg.MODEL.ENCODER_DROPOUT,
        encoder_weights=encoder_weights,
    )
