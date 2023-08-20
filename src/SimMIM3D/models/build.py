from .pretrain_simmim import PretrainSimMIM
from .finetune_unetr import FinetuneUNETR

def build_model(config, net, is_pretrain=True):
    if is_pretrain:
        return PretrainSimMIM(
            net = net,
            learning_rate=config.TRAINING.BASE_LR,
            weight_decay=config.TRAINING.WEIGHT_DECAY,
            lr_warmup_epochs=config.TRAINING.LR_WARMUP_EPOCHS,
            epochs=config.TRAINING.EPOCHS,
        )
    else:
        return FinetuneUNETR(
            net = net,
            learning_rate=config.TRAINING.BASE_LR,
            weight_decay=config.TRAINING.WEIGHT_DECAY,
            lr_warmup_epochs=config.TRAINING.LR_WARMUP_EPOCHS,
            freeze_warmup_epochs=config.TRAINING.FREEZE_WARMUP_EPOCHS,
            epochs=config.TRAINING.EPOCHS,
            num_classes=config.DATA.NUM_CLASSES,
        )