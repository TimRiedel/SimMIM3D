import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from src.models.SimMIM import MAE, SimMIM3D, BratsImageModule 
from src.models.SimMIM.config import get_config, convert_cfg_to_dict

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

if __name__ == "__main__":
    cfg = get_config()
    cfg.freeze()
    print(cfg)

    if torch.cuda.get_device_name() == "NVIDIA A40":
        torch.set_float32_matmul_precision('medium')

    network = SimMIM3D(
        img_size=cfg.DATA.IMG_SIZE,
        in_channels=cfg.MODEL.IN_CHANNELS,
        patch_size=cfg.MODEL.PATCH_SIZE,
        dropout_rate=cfg.MODEL.ENCODER_DROPOUT,
    )
    
    model = MAE(
        net = network,
        learning_rate=cfg.TRAINING.BASE_LR,
        optimizer_class=torch.optim.AdamW,
        weight_decay=cfg.TRAINING.WEIGHT_DECAY,
        warmup_epochs=cfg.TRAINING.WARMUP_EPOCHS,
        epochs=cfg.TRAINING.EPOCHS,
    )
    
    data = BratsImageModule(
        data_dir=cfg.DATA.BRATS_DATA_DIR,
        img_size=cfg.DATA.IMG_SIZE,
        patch_size=cfg.MODEL.PATCH_SIZE,
        mask_ratio=cfg.DATA.MASK_RATIO,
        batch_size=cfg.DATA.BATCH_SIZE, 
        num_workers=cfg.DATA.NUM_WORKERS
    )

    wandb_config = convert_cfg_to_dict(cfg)
    wandb_config.pop("LOGGING")
    wandb_config["SYSTEM"] = {
        "GPU-Type": torch.cuda.get_device_name(),
        "GPU-Count": torch.cuda.device_count(),
        "NUM_WORKERS": cfg.DATA.NUM_WORKERS,
    }

    logger = WandbLogger(
        project="ba-thesis",
        name=cfg.LOGGING.RUN_NAME,
        dir=cfg.LOGGING.WANDB_DIR,
        config=wandb_config,
        log_model="all",
    )

    callbacks = [
        # EarlyStopping(monitor="validation/loss", patience=20),
        ModelCheckpoint(dirpath=cfg.LOGGING.CHECKPOINT_DIR, monitor="validation/loss")
    ]

    trainer = pl.Trainer(
        # Compute
        accelerator=cfg.SYSTEM.ACCELERATOR, 
        strategy=cfg.SYSTEM.STRATEGY,
        devices=cfg.SYSTEM.DEVICES,
        num_nodes=cfg.SYSTEM.NODES,
        precision=cfg.SYSTEM.PRECISION, 

        # Training
        max_epochs=cfg.TRAINING.EPOCHS,

        # Logging
        callbacks=callbacks,
        logger=logger,
        profiler="simple",
    )

    trainer.fit(model, data)