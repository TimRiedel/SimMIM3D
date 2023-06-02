import torch
import pytorch_lightning as pl
import torchmetrics
import wandb
from monai.metrics import PSNRMetric
from monai.optimizers import WarmupCosineSchedule

class MAE(pl.LightningModule):
    def __init__(
            self,
            net,
            loss_fn,
            learning_rate:
            float, optimizer_class,
            weight_decay: float,
            warmup_epochs: int,
            epochs: int
        ):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs

        # Metrics
        # self.psnr = PSNRMetric()

        # Logging
        self.save_hyperparameters(ignore=["net", "loss_fn", "learning_rate"])
        self.num_samples = 10
        self.channel_idx = 0

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )

        lr_scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=self.warmup_epochs,
            t_total=self.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def prepare_batch(self, batch):
        return batch["image"], batch["mask"]

    def common_step(self, x, mask, batch_idx):
        x_pred = self.net(x, mask)
        loss = self.loss_fn(x_pred, x)
        return loss, x_pred


    # Training
    def training_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_pred = self.common_step(x, mask, batch_idx)
        return {"loss": loss, "x_pred": x_pred}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("training/loss", outputs["loss"], on_epoch=True, sync_dist=True, batch_size=batch["image"].shape[0]) # type: ignore
    

    # Validation
    def validation_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_pred = self.common_step(x, mask, batch_idx)
        return {"loss": loss, "x_pred": x_pred}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log("validation/loss", outputs["loss"], on_step=True, sync_dist=True, batch_size=batch["image"].shape[0])

        if batch_idx == 0:
            table = wandb.Table(columns=["original", "reconstruction"])
            images = batch["image"][:self.num_samples, self.channel_idx].detach().cpu()
            reconstructions = outputs["x_pred"][:self.num_samples, self.channel_idx].detach().cpu()
            slice_idx = images[0].shape[2] // 2

            for i in range(images.shape[0]):
                img = images[i, :, :, slice_idx]
                recon = reconstructions[i, :, :, slice_idx]
                table.add_data(wandb.Image(img), wandb.Image(recon))

            self.logger.experiment.log({"validation/images": table}) # type: ignore


    # Testing
    def test_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_pred = self.common_step(x, mask, batch_idx)
        return {"loss": loss, "x_pred": x_pred}

    
    # Prediction
    def predict_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        prediction = self.net(x)
        return prediction