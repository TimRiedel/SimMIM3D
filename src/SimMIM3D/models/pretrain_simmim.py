import pytorch_lightning as pl
import wandb
from monai.metrics import PSNRMetric
from monai.optimizers import WarmupCosineSchedule

class PretrainSimMIM(pl.LightningModule):
    def __init__(
            self,
            net,
            learning_rate: float, 
            optimizer_class,
            weight_decay: float,
            warmup_epochs: int,
            epochs: int
        ):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs

        # Logging
        self.save_hyperparameters(ignore=["net", "learning_rate"])
        self.num_samples = 10
        self.channel_idx = 0

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )

        return {"optimizer": optimizer}
        # lr_scheduler = WarmupCosineSchedule(
        #     optimizer=optimizer,
        #     warmup_steps=self.warmup_epochs,
        #     t_total=self.epochs + self.warmup_epochs
        # )
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}

    def prepare_batch(self, batch):
        return batch["image"], batch["mask"]


    # Training
    def training_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_pred, _ = self.net(x, mask)
        return {"loss": loss, "x_pred": x_pred}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("training/loss", outputs["loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["image"].shape[0]) # type: ignore
    

    # Validation
    def validation_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_pred, x_masked = self.net(x, mask)
        return {"loss": loss, "x_pred": x_pred, "x_masked": x_masked}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log("validation/loss", outputs["loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["image"].shape[0])

        if batch_idx == 0:
            slice_idx = batch["image"].shape[4] // 2

            if self.current_epoch == 0:
                images = batch["image"][:self.num_samples, self.channel_idx, :, :, slice_idx].detach().cpu()
                self.logger.experiment.log({"validation/original": [wandb.Image(img) for img in images]}) # type: ignore

            masked = outputs["x_masked"][:self.num_samples, self.channel_idx, :, :, slice_idx].detach().cpu()
            reconstructions = outputs["x_pred"][:self.num_samples, self.channel_idx, :, :, slice_idx].detach().cpu()
            self.logger.experiment.log({"validation/masked": [wandb.Image(mask) for mask in masked]}) # type: ignore
            self.logger.experiment.log({"validation/reconstruction": [wandb.Image(recon) for recon in reconstructions]}) # type: ignore


    # Testing
    def test_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_pred = self.net(x, mask)
        return {"loss": loss, "x_pred": x_pred}

    
    # Prediction
    def predict_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_pred = self.net(x, mask)
        return x_pred

