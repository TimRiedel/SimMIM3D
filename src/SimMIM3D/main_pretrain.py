import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from src.SimMIM3D.networks import SimMIM3D
from src.SimMIM3D.data import BratsImageModule
from src.SimMIM3D.models import PretrainSimMIM
from src.SimMIM3D.config import get_config, convert_cfg_to_dict

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

if __name__ == "__main__":
    cfg = get_config()
    cfg.freeze()
    print(f"\n{cfg}\n")

    if torch.cuda.get_device_name() == "NVIDIA A40":
        torch.set_float32_matmul_precision('medium')

    for k in range(cfg.TRAINING.CROSS_VALIDATIONS):
        network = SimMIM3D(
            img_size=cfg.DATA.IMG_SIZE,
            in_channels=cfg.MODEL.IN_CHANNELS,
            patch_size=cfg.MODEL.PATCH_SIZE,
            dropout_rate=cfg.MODEL.ENCODER_DROPOUT,
        )
        
        model = PretrainSimMIM(
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

        run_name = f"{cfg.LOGGING.RUN_NAME}_v{cfg.LOGGING.VERSION}"
        wandb_log_dir = f"{cfg.LOGGING.JOBS_DIR}/logs/wandb"
        ckpt_dir = f"{cfg.LOGGING.JOBS_DIR}/checkpoints/{run_name}"
        if cfg.TRAINING.CROSS_VALIDATIONS > 1:
            run_name = f"{run_name}_cv_{k}"
            ckpt_dir = f"{ckpt_dir}/cv_{k}"

        wandb_config = convert_cfg_to_dict(cfg)
        wandb_config.pop("LOGGING")
        wandb_config["SYSTEM"] = {
            "GPU-Type": torch.cuda.get_device_name(),
            "GPU-Count": torch.cuda.device_count(),
            "NUM_WORKERS": cfg.DATA.NUM_WORKERS,
        }

        logger = WandbLogger(
            project="ba-thesis",
            name=run_name,
            dir=wandb_log_dir,
            config=wandb_config,
            log_model="all",
        )

        callbacks = [
            ModelCheckpoint(dirpath=ckpt_dir, monitor="validation/loss")
        ]

        trainer = pl.Trainer(
            # Compute
            accelerator=cfg.SYSTEM.ACCELERATOR, 
            strategy=cfg.SYSTEM.STRATEGY,
            devices=cfg.SYSTEM.DEVICES,
            num_nodes=cfg.SYSTEM.NODES,
            precision=cfg.SYSTEM.PRECISION, 

            # Training
            max_epochs=cfg.TRAINING.EPOCHS + cfg.TRAINING.WARMUP_EPOCHS,

            # Logging
            callbacks=callbacks,
            logger=logger,
            profiler="simple",
        )

        trainer.fit(model, data)