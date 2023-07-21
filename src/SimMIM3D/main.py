import torch
import wandb
import warnings
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from src.SimMIM3D.config import get_config, convert_cfg_to_dict
from src.SimMIM3D.networks import build_network
from src.SimMIM3D.models import build_model
from src.SimMIM3D.data import build_data
from src.mlutils.callbacks import LogBratsValidationPredictions

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def main(
        config, 
        dataset="adni",
        is_pretrain=True,
        dev_run=False
    ):
    for cv in range(config.TRAINING.CV_FOLDS):
        pl.seed_everything(1209 + cv**3) # make it reproducible but still kind of random

        data = build_data(config, is_pretrain=is_pretrain, dataset=dataset)
        network = build_network(config, is_pretrain=is_pretrain)
        model = build_model(config, network, is_pretrain=is_pretrain)

        wandb_log_dir = f"{config.LOGGING.JOBS_DIR}/logs/"
        run_name = config.LOGGING.RUN_NAME
        if config.TRAINING.CV_FOLDS > 1:
            run_name = f"{run_name}_cv{cv}"
        ckpt_dir = f"{config.LOGGING.CKPT_DIR}/{run_name}"

        wandb_config = convert_cfg_to_dict(config)
        wandb_config.pop("LOGGING")

        logger = None
        if not dev_run:
            logger = WandbLogger(
                project=config.LOGGING.PROJECT_NAME,
                name=run_name,
                save_dir=wandb_log_dir,
                config=wandb_config,
                log_model="True",
            )

        callbacks: list[pl.Callback] = [
            ModelCheckpoint(dirpath=ckpt_dir, monitor="validation/loss"),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        if not is_pretrain and dataset == "brats":
            callbacks.append(LogBratsValidationPredictions(num_samples=config.DATA.BATCH_SIZE))
            callbacks.append(EarlyStopping(monitor="validation/loss", mode="min", patience=50, min_delta=0.007))


        trainer = pl.Trainer(
            # Compute
            accelerator=config.SYSTEM.ACCELERATOR, 
            strategy=config.SYSTEM.STRATEGY,
            devices=config.SYSTEM.DEVICES,
            num_nodes=config.SYSTEM.NODES,
            precision=config.SYSTEM.PRECISION, 

            # Training
            max_epochs=config.TRAINING.EPOCHS + config.TRAINING.LR_WARMUP_EPOCHS,
            fast_dev_run=dev_run,

            # Logging
            callbacks=callbacks,
            logger=logger,
            profiler="simple",
            log_every_n_steps=config.DATA.BATCH_SIZE,
        )

        trainer.fit(model, data)
        wandb.finish()


def parse_options():
    parser = argparse.ArgumentParser('SimMIM3D script', add_help=False)

    parser.add_argument('--finetune', action='store_true', help="finetune only the decoder without pre-training")
    parser.add_argument('--quick', action='store_true', help="quick run for debugging purposes")
    parser.add_argument('--dataset', type=str, choices=['brats', 'adni'], default='adni', help="dataset to use (brats or adni)")
    parser.add_argument('--lr', type=float, help="learning rate for training")
    parser.add_argument('--mask_ratio', type=float, help="ratio of masked patches for pre-training")
    parser.add_argument('--train_frac', type=float, help="fraction of training data for finetuning")
    parser.add_argument('--name_suffix', type=str, default="", help="suffix for run name")
    parser.add_argument('--load_checkpoint', type=str, help="path to checkpoint relative to checkpoint folder to use for finetuning")
    parser.add_argument('--cv', type=int, help="number of cross validation folds to run")

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

    main(
        config=config, 
        dataset=args.dataset, 
        is_pretrain=is_pretrain, 
        dev_run=args.quick
    )