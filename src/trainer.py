import pytorch_lightning as pl


def build_trainer(config, dev_run, callbacks, logger):
    return pl.Trainer(
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
