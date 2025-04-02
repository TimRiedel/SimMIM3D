from pytorch_lightning.loggers import WandbLogger

from src.utils.config_to_dict import convert_cfg_to_dict


def build_logger(config, dev_run, cv=None):
    run_name = config.LOGGING.RUN_NAME
    if config.TRAINING.CV_FOLDS > 1 and cv is not None:
        run_name = f"{run_name}_cv{cv}"

    ckpt_dir = f"{config.LOGGING.CKPT_DIR}/{run_name}"
    wandb_config = convert_cfg_to_dict(config)
    wandb_config.pop("LOGGING")

    logger = None
    if not dev_run:
        logger = WandbLogger(
            project=config.LOGGING.PROJECT_NAME,
            name=run_name,
            save_dir=config.LOGGING.LOGS_DIR,
            config=wandb_config,
            log_model=True,
        )
    return ckpt_dir, logger
