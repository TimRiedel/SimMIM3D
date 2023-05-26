import torch
import pytorch_lightning as pl
import torchmetrics

class MAE(pl.LightningModule):
    def __init__(self, net, loss_fn, learning_rate, optimizer_class):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.optimizer_class = optimizer_class
        self.train_step_outputs = []

        # TODO: add metrics
        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def configure_optimizers(self):
        # TODO: add learning rate scheduler
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def common_step(self, batch, batch_idx):
        x = batch["image"]
        x_pred = self.net(x)
        loss = self.loss_fn(x_pred, x)
        return loss, x_pred

    def training_step(self, batch, batch_idx):
        loss, x_pred = self.common_step(batch, batch_idx)
        # TODO: add metrics
        return {"loss": loss, "x_pred": x_pred}
    
    def validation_step(self, batch, batch_idx):
        loss, x_pred = self.common_step(batch, batch_idx)
        # TODO: add metrics
        return {"loss": loss, "x_pred": x_pred}

    def test_step(self, batch, batch_idx):
        loss, x_pred = self.common_step(batch, batch_idx)
        # TODO: add metrics
        return {"loss": loss, "x_pred": x_pred}

    def predict_step(self, batch, batch_idx):
        x = self.prepare_batch(batch)
        prediction = self.net(x)
        return prediction