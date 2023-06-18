import pytorch_lightning as pl
from torchmetrics import Accuracy, Dice
from monai.optimizers import WarmupCosineSchedule
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

class FinetuneUNETR(pl.LightningModule):
    def __init__(
            self,
            net,
            learning_rate: float, 
            optimizer_class,
            weight_decay: float,
            warmup_epochs: int,
            epochs: int,
            num_classes: int,
        ):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.num_classes = num_classes

        self.loss_fn = DiceLoss(
            squared_pred=True, 
            softmax=True,
        )

        self.train_accuracy = Accuracy(task="multilabel", num_labels=self.num_classes)
        self.val_accuracy = Accuracy(task="multilabel", num_labels=self.num_classes)

        self.train_dice_score = DiceMetric(include_background=True, num_classes=self.num_classes, reduction="mean_batch")
        self.val_dice_score = DiceMetric(include_background=True, num_classes=self.num_classes, reduction="mean_batch")

        # Logging
        self.save_hyperparameters(ignore=["net", "learning_rate", "loss_fn"])


    def configure_optimizers(self):
        # TODO: Consider layer-wise learning rate decay (layer decay ratio = 0.75) to
        # stabilize the ViT training as suggested in "Self-Pretraining with MAE"
        optimizer = self.optimizer_class(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )

        lr_scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=self.warmup_epochs,
            t_total=self.epochs + self.warmup_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}

    def prepare_batch(self, batch):
        return batch["image"], batch["label"]

    def common_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred


    # Training
    def training_step(self, batch, batch_idx):
        loss, y_pred = self.common_step(batch, batch_idx) 
        return {"loss": loss, "y_pred": y_pred}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.train_accuracy(outputs["y_pred"], batch["label"]) # type: ignore
        self.train_dice_score(outputs["y_pred"], batch["label"]) # type: ignore

        self.log("training/loss", outputs["loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["image"].shape[0]) # type: ignore
        self.log("training/accuracy", self.train_accuracy, on_step=False, on_epoch=True, sync_dist=True)

    def on_train_epoch_end(self):
        dice = self.train_dice_score.aggregate()
        for i in range(dice.shape[0]): # type: ignore
            self.log(f"training/dice_score_{i}", dice[i], on_step=False, on_epoch=True, sync_dist=True) # type: ignore
        self.train_dice_score.reset()


    # Validation
    def validation_step(self, batch, batch_idx):
        loss, y_pred = self.common_step(batch, batch_idx) 
        return {"loss": loss, "y_pred": y_pred}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.val_accuracy(outputs["y_pred"], batch["label"])
        self.val_dice_score(outputs["y_pred"], batch["label"])

        self.log("validation/loss", outputs["loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["image"].shape[0])
        self.log("validation/accuracy", self.val_accuracy, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        dice = self.val_dice_score.aggregate()
        for i in range(dice.shape[0]): # type: ignore
            self.log(f"validation/dice_score_{i}", dice[i], on_step=False, on_epoch=True, sync_dist=True) # type: ignore
        self.val_dice_score.reset()
    

    # Testing
    def test_step(self, batch, batch_idx):
        loss, y_pred = self.common_step(batch, batch_idx) 
        return {"loss": loss, "y_pred": y_pred}
