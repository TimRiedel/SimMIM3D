import pytorch_lightning as pl
import torchmetrics
import wandb
from monai.metrics import PSNRMetric
from monai.optimizers import WarmupCosineSchedule

class FinetuneUNETR(pl.LightningModule):
    def __init__(
            self,
            net,
            loss_fn,
            learning_rate: float, 
            optimizer_class,
            weight_decay: float,
            warmup_epochs: int,
            epochs: int,
            num_classes: int,
        ):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.num_classes = num_classes

        self.train_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=self.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=self.num_classes)
        self.train_dice_score = torchmetrics.Dice(task="multilabel", num_labels=self.num_classes)
        self.val_dice_score = torchmetrics.Dice(task="multilabel", num_labels=self.num_classes)

        # Logging
        self.save_hyperparameters(ignore=["net", "learning_rate", "loss_fn"])
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
        return batch["image"], batch["label"]

    def common_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y


    # Training
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx) 
        return {"loss": loss, "y_pred": y_pred}
    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     self.log("training/loss", outputs["loss"], on_epoch=True, sync_dist=True, batch_size=batch["image"].shape[0]) # type: ignore
    

    # Validation
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx) 
        return {"loss": loss, "y_pred": y_pred}

    # def on_validation_batch_end(self, outputs, batch, batch_idx):
    #     self.log("validation/loss", outputs["loss"], on_step=True, sync_dist=True, batch_size=batch["image"].shape[0])

    #     if batch_idx == 0:
    #         slice_idx = batch["image"].shape[4] // 2
    #         images = batch["image"][:self.num_samples, self.channel_idx, :, :, slice_idx].detach().cpu()
    #         reconstructions = outputs["x_pred"][:self.num_samples, self.channel_idx, :, :, slice_idx].detach().cpu()

    #         self.logger.experiment.log({"validation/original": [wandb.Image(img) for img in images]})
    #         self.logger.experiment.log({"validation/reconstruction": [wandb.Image(recon) for recon in reconstructions]})


    # Testing
    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx) 
        return {"loss": loss, "y_pred": y_pred}
