import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from monai.networks.nets import UNet

from .config import *
from .model import UNet3D

if __name__ == "__main__":
    network = UNet(
        spatial_dims=3,
        in_channels=4, # 4 modalities (FLAIR, T1, T1ce, T2)
        out_channels=4, # 4 segmentation classes
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2, #TODO: Understand
        dropout=0.1, #TODO: Understand
    )

    model = UNet3D(
        net=network,
        num_classes=NUM_CLASSES,
        loss_fn=None, #TODO: Choose loss function
        learning_rate=LEARNING_RATE, #TODO: Choose learning rate
        optimizer_class=torch.optim.AdamW, #TODO: Choose optimizer
    )

    #TODO: import data module
    data = BraTSDataModule(
        data_dir=DATA_DIR, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(monitor="val_accuracy", mode="max"),
    ]

    logger = WandbLogger(project="ba-thesis", name="Unet3D")

    #TODO: Add profiler

    trainer = pl.Trainer(
        accelerator=ACCELERATOR, 
        devices=DEVICES, 
        strategy="ddp",
        precision=PRECISION, 
        max_epochs=NUM_EPOCHS,
        callbacks=callbacks,
        logger=logger,
        profiler="simple",
    )

    trainer.fit(model, data)
