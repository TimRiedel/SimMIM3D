import numpy as np
import wandb
import pytorch_lightning as pl

class LogValidationPredictions(pl.Callback):
    def __init__(self, num_samples=5, channel_idx=0):
        super().__init__()
        self.num_samples = num_samples
        self.channel_idx = channel_idx
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):
        """Logs validation images and predictions for the first batch and
        the given channel index. Assumes outputs has the shape {loss, y_pred, y}."""

        if batch_idx == 0:
            n = self.num_samples
            x, y = batch["image"], batch["label"]
            y_preds = outputs["y_pred"]

            images = x[:n, self.channel_idx]
            true_labels = y[:n]
            pred_labels = y_preds[:n]

            slice_idx = images[0].shape[2] // 2

            columns = ["Image", "Ground Truth", "Prediction"]
            data = []
            for img, truth, pred in zip(images, true_labels, pred_labels):
                img = img.cpu()
                truth = np.argmax(truth.cpu(), axis=0)
                pred = np.argmax(pred.cpu(), axis=0)

                data.append([
                    wandb.Image(img[:,:,slice_idx]),
                    wandb.Image(truth[:,:,slice_idx]), 
                    wandb.Image(pred[:,:,slice_idx]),
                ])

            trainer.logger.log_table(key=f"validation/images/channel_{self.channel_idx}", columns=columns, data=data)