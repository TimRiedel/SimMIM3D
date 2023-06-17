import torch

from .pretrain_simmim import PretrainSimMIM
from .finetune_unetr import FinetuneUNETR

def build_model(config, net, is_pretrain=True):
    if is_pretrain:
        return PretrainSimMIM(
            net = net,
            learning_rate=config.TRAINING.BASE_LR,
            optimizer_class=torch.optim.AdamW,
            weight_decay=config.TRAINING.WEIGHT_DECAY,
            warmup_epochs=config.TRAINING.WARMUP_EPOCHS,
            epochs=config.TRAINING.EPOCHS,
        )
    else:
        return FinetuneUNETR(
            net = net,
            learning_rate=config.TRAINING.BASE_LR,
            optimizer_class=torch.optim.AdamW,
            weight_decay=config.TRAINING.WEIGHT_DECAY,
            warmup_epochs=config.TRAINING.WARMUP_EPOCHS,
            epochs=config.TRAINING.EPOCHS,
            num_classes=config.DATA.NUM_CLASSES,
        )