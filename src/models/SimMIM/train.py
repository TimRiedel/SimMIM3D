import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from src.models.SimMIM import MAE, SimMIM3D, BratsImageModule 
from src.models.SimMIM.config import *

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

if __name__ == "__main__":
    network = SimMIM3D(
        img_size=INPUT_SIZE,
        in_channels=NUM_CHANNELS,
        # TODO: higher patch size reduces computation needs
        patch_size=PATCH_SIZE,
    )
    
    # TODO: observe if reduction="none" is necessary
    loss_fn = nn.MSELoss() 

    model = MAE(
        net = network,
        loss_fn=loss_fn,
        num_channels=NUM_CHANNELS,
        learning_rate=LEARNING_RATE,
        optimizer_class=torch.optim.AdamW,
        # TODO: add learning rate scheduler
    )
    
    data = BratsImageModule(
        data_dir=BRATS_DATA_DIR, 
        batch_size=BATCH_SIZE, 
        input_size=INPUT_SIZE,
        num_workers=NUM_WORKERS
    )

    logger = WandbLogger(project="ba-thesis", name=RUN_NAME, log_model="all", dir=WANDB_DIR)

    callbacks = ModelCheckpoint(dirpath=CHECKPOINT_DIR, monitor="validation/loss")

    trainer = pl.Trainer(
        # Compute
        accelerator=ACCELERATOR, 
        strategy="ddp",
        devices=DEVICES,
        num_nodes=NODES,
        precision=PRECISION, 

        # Training
        max_epochs=NUM_EPOCHS,

        # Logging
        callbacks=callbacks,
        logger=logger,
        profiler="simple",
    )

    trainer.fit(model, data)