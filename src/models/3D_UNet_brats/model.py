import torch
import pytorch_lightning as pl
import torchmetrics

class Unet3D(pl.LightningModule):
    def __init__(self, net, loss_fn, learning_rate, num_classes):
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes) #TODO: Choose multiclass or multilabel
        self.dice_score = torchmetrics.Dice(task="multiclass", num_classes=self.num_classes) #TODO: Choose multiclass or multilabe

    def configure_optimizers(self):
        #TODO: Choose Scheduler
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        #TODO: Implement
        # return batch["image"][tio.DATA], batch["label"][tio.DATA]
        pass

    def common_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y,

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        dice_score = self.dice_score(y_pred, y)
        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy, "train_dice_score": dice_score}, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x = self.prepare_batch(batch)
        y_pred = self.net(x)
        prediction = torch.argmax(y_pred, dim=1) #TODO: Understand
        return prediction