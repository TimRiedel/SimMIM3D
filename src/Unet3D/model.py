import torch
import pytorch_lightning as pl
import torchmetrics

class Unet3D(pl.LightningModule):
    def __init__(self, net, loss_fn, learning_rate, num_classes, optimizer_class):
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.optimizer_class = optimizer_class
        self.train_step_outputs = []

        self.train_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=self.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=self.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=self.num_classes)
        self.dice_score = torchmetrics.Dice(task="multilabel", num_labels=self.num_classes)

        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def configure_optimizers(self):
        #TODO: Choose Scheduler
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch["image"], batch["label"]

    def common_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)
        self.train_step_outputs.append({"y_pred": y_pred, "y": y})
        self.log( "training/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "y_pred": y_pred, "y": y}
    
    def on_train_epoch_end(self):
        y_pred = torch.cat([labels["y_pred"] for labels in self.train_step_outputs])
        y = torch.cat([labels["y"] for labels in self.train_step_outputs])
        self.log_dict(
            {
                "training/accuracy": self.train_accuracy(y_pred, y),
                "training/dice_score": self.dice_score(y_pred, y),
            },
            on_epoch=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss, on_epoch=True, sync_dist=True)
        return {"loss": loss, "y_pred": y_pred, "y": y}

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        return {"loss": loss, "y_pred": y_pred, "y": y}

    def predict_step(self, batch, batch_idx):
        x = self.prepare_batch(batch)
        y_pred = self.net(x)
        prediction = torch.argmax(y_pred, dim=1) #TODO: Understand
        return prediction