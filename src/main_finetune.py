import torch
import wandb
import warnings
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from src.configs.config import get_config
from src.logger import build_logger
from src.networks import build_network
from src.models import build_model
from src.data import build_data
from src.trainer import build_trainer
from src.utils.log_brats_validation_predictions import LogBratsValidationPredictions

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def main(
        config,
        dataset="brats",
        dev_run=False
):
    for cv in range(config.TRAINING.CV_FOLDS):
        pl.seed_everything(1209 + cv ** 3)  # make it reproducible but still kind of random

        data = build_data(config, dataset=dataset, is_pretrain=False)
        network = build_network(config, is_pretrain=False)
        model = build_model(config, network, is_pretrain=False)

        ckpt_dir, logger = build_logger(config, dev_run, cv)

        callbacks: list[pl.Callback] = [
            ModelCheckpoint(dirpath=ckpt_dir, monitor="validation/loss"),
            LearningRateMonitor(logging_interval="epoch"),
            LogBratsValidationPredictions(num_samples=config.DATA.BATCH_SIZE),
            EarlyStopping(monitor="validation/loss", mode="min", patience=80, min_delta=0.005)
        ]

        trainer = build_trainer(config, dev_run, callbacks, logger)
        trainer.fit(model, data)
        wandb.finish()


def parse_options():
    parser = argparse.ArgumentParser('Finetune script', add_help=True)

    parser.add_argument('--project_name', type=str, help="Project name for logging.")
    parser.add_argument('--run_name', type=str, default="", help="Run name for logging.")
    parser.add_argument('--ckpt_dir', type=str, help="Directory under which to save checkpoints. A new folder with "
                                                     "the run name will be created in this directory.")
    parser.add_argument('--logs_dir', type=str, help="Directory under which to store logs. A new folder with the run "
                                                     "name will be created in this directory.")
    parser.add_argument('--dev_run', action='store_true', help="Quick run for debugging purposes.")
    parser.add_argument('--lr', type=float, help="Base learning rate for training.")
    parser.add_argument('--train_frac', type=float, help="Fraction of training data to use for fine-tuning.")
    parser.add_argument('--load_model', type=str,
                        help="Path to model checkpoint to use for fine-tuning.")
    parser.add_argument('--cv', type=int, help="Number of cross validation folds to run.")

    arguments = parser.parse_args()

    config = get_config(arguments, pre_training=False)
    config.freeze()
    print(f"\n{config}\n")

    return arguments, config


if __name__ == "__main__":
    args, cfg = parse_options()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    main(
        config=cfg,
        dev_run=args.dev_run
    )
