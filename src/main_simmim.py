import torch
import wandb
import warnings
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.configs.config import get_config
from src.networks import build_network
from src.models import build_model
from src.data import build_data

from src.logger import build_logger
from src.trainer import build_trainer

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def main(
        config,
        dataset="adni",
        dev_run=False
):
    pl.seed_everything(1209)

    data = build_data(config, dataset=dataset, is_pretrain=True)
    network = build_network(config, is_pretrain=True)
    model = build_model(config, network, is_pretrain=True)

    ckpt_dir, logger = build_logger(config, dev_run)

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(dirpath=ckpt_dir, monitor="validation/loss"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = build_trainer(config, dev_run, callbacks, logger)

    trainer.fit(model, data)
    wandb.finish()


def parse_options():
    parser = argparse.ArgumentParser('Pre-Training with SimMIM', add_help=True)

    parser.add_argument('--project_name', type=str, help="Project name for logging.")
    parser.add_argument('--run_name', type=str, default="", help="Run name for logging.")
    parser.add_argument('--ckpt_dir', type=str, help="Directory under which to save checkpoints. A new folder with "
                                                     "the run name will be created in this directory.")
    parser.add_argument('--logs_dir', type=str, help="Directory under which to store logs. A new folder with the run "
                                                     "name will be created in this directory.")
    parser.add_argument('--dev_run', action='store_true', help="Quick run for debugging purposes.")
    parser.add_argument('--dataset', type=str, choices=['brats', 'adni'], default='adni',
                        help="Dataset to use for pre-training (BraTS or ADNI).")
    parser.add_argument('--lr', type=float, help="Base learning rate for training.")
    parser.add_argument('--mask_ratio', type=float, help="ratio of masked patches to visible patches for pre-training.")

    arguments = parser.parse_args()

    config = get_config(arguments, pre_training=True)
    config.freeze()
    print(f"\n{config}\n")

    return arguments, config


if __name__ == "__main__":
    args, cfg = parse_options()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    main(
        config=cfg,
        dataset=args.dataset,
        dev_run=args.dev_run
    )
