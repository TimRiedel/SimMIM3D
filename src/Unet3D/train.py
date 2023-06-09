import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from monai.losses import DiceLoss
from monai.networks.nets import UNet

from src.Unet3D.config import *
from src.Unet3D import BratsDataModule, Unet3D
from src.mlutils.callbacks import LogValidationPredictions

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

if __name__ == "__main__":
    pl.seed_everything(1)

    network = UNet(
        spatial_dims=3,
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CLASSES,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,
        dropout=0.1,
    )
    
    loss_fn = DiceLoss(
        squared_pred=True, 
        to_onehot_y=False, 
        softmax=True,
    )

    model = Unet3D(
        net=network,
        num_classes=NUM_CLASSES,
        loss_fn=loss_fn,
        learning_rate=LEARNING_RATE,
        optimizer_class=torch.optim.AdamW,
    )
    

    data = BratsDataModule(
        data_dir=DATA_DIR, 
        batch_size=BATCH_SIZE, 
        input_size=INPUT_SIZE,
        num_workers=NUM_WORKERS
    )

    callbacks = [
        # EarlyStopping(monitor="val_loss", patience=5),
        # ModelCheckpoint(monitor="val_accuracy", mode="max"),
        LogValidationPredictions(),
    ]

    logger = WandbLogger(project="ba-thesis", name="Unet3D Brats", log_model="all")

    #TODO: Add profiler

    trainer = pl.Trainer(
        accelerator=ACCELERATOR, 
        devices=DEVICES,
        num_nodes=2,
        strategy="ddp",
        precision=PRECISION, 
        max_epochs=NUM_EPOCHS,
        # TODO: Add gradient accumulation
        # accumulate_grad_batches=BATCH_SIZE
        callbacks=callbacks,
        logger=logger,
        profiler="simple",
        num_sanity_val_steps=2,
        limit_train_batches=2,
        limit_val_batches=2
    )

    trainer.fit(model, data)