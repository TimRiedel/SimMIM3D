import torch
from monai.losses import DiceLoss

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
        loss_fn = DiceLoss(
            squared_pred=True, 
            to_onehot_y=False, 
            softmax=True,
        )

        return FinetuneUNETR(
            net = net,
            loss_fn = loss_fn,
            learning_rate=config.TRAINING.BASE_LR,
            optimizer_class=torch.optim.AdamW,
            weight_decay=config.TRAINING.WEIGHT_DECAY,
            warmup_epochs=config.TRAINING.WARMUP_EPOCHS,
            epochs=config.TRAINING.EPOCHS,
            num_classes=config.DATA.NUM_CLASSES,
        )