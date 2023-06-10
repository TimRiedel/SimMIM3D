import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from src.SimMIM3D.config import get_config, convert_cfg_to_dict
from src.SimMIM3D.networks import build_network
from src.SimMIM3D.models import build_model
from src.SimMIM3D.data import build_data

import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def main(config, is_pretrain=True):
    for k in range(config.TRAINING.CROSS_VALIDATIONS):
        network = build_network(config, is_pretrain=is_pretrain)
        model = build_model(config, network, is_pretrain=is_pretrain)
        data = build_data(config, is_pretrain=is_pretrain)

        run_name = f"{config.LOGGING.RUN_NAME}_v{config.LOGGING.VERSION}"
        wandb_log_dir = f"{config.LOGGING.JOBS_DIR}/logs/wandb"
        ckpt_dir = f"{config.LOGGING.JOBS_DIR}/checkpoints/{run_name}"
        if config.TRAINING.CROSS_VALIDATIONS > 1:
            run_name = f"{run_name}_cv_{k}"
            ckpt_dir = f"{ckpt_dir}/cv_{k}"

        wandb_config = convert_cfg_to_dict(config)
        wandb_config.pop("LOGGING")
        wandb_config["SYSTEM"] = {
            "GPU-Type": torch.cuda.get_device_name(),
            "GPU-Count": torch.cuda.device_count(),
            "NUM_WORKERS": config.DATA.NUM_WORKERS,
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
            accelerator=config.SYSTEM.ACCELERATOR, 
            strategy=config.SYSTEM.STRATEGY,
            devices=config.SYSTEM.DEVICES,
            num_nodes=config.SYSTEM.NODES,
            precision=config.SYSTEM.PRECISION, 

            # Training
            max_epochs=config.TRAINING.EPOCHS + config.TRAINING.WARMUP_EPOCHS,
            fast_dev_run=True,

            # Logging
            callbacks=callbacks,
            logger=logger,
            profiler="simple",
        )

        trainer.fit(model, data)


def parse_options():
    parser = argparse.ArgumentParser('SimMIM3D script', add_help=False)

    parser.add_argument('--finetune', action='store_true', help="finetune only the decoder without pre-training")

    args = parser.parse_args()

    config = get_config(args)
    config.freeze()
    print(f"\n{config}\n")

    return args, config


if __name__ == "__main__":
    args, config = parse_options()
    is_pretrain = not args.finetune


    if torch.cuda.get_device_name() == "NVIDIA A40":
        torch.set_float32_matmul_precision('medium')

    main(config, is_pretrain)