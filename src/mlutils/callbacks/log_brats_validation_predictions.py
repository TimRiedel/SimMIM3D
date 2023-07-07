import numpy as np
import torch
import wandb
import pytorch_lightning as pl

class LogBratsValidationPredictions(pl.Callback):
    def __init__(
            self, 
            num_samples, 
            class_labels={1: "ED", 2: "NET", 3: "ET"}
        ):
        super().__init__()
        self.num_samples = num_samples
        self.class_labels = class_labels
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):
        """Logs validation images and predictions for the first batch and
        the given channel index. Assumes outputs has a field 'y_pred'.
        Labels must not be in one-hot encoding."""

        if batch_idx != 0:
            return

        slice_idx = batch["image"].shape[4] // 2

        images = batch["image"][:self.num_samples, :, :, :, slice_idx].detach().cpu()
        images = images[:, 3] + images[:, 2]

        labels = batch["label"][:self.num_samples, 0, :, :, slice_idx].detach().cpu()
        preds = outputs["y_pred"][:self.num_samples, 0, :, :, slice_idx].detach().cpu()
        self.log_images(images, labels, preds, modality="t2_t1gd", class_labels=self.class_labels, trainer=trainer)


    def log_images(
        self, 
        images: np.ndarray, 
        labels: np.ndarray, 
        preds: np.ndarray, 
        modality: str, 
        class_labels: dict, 
        trainer: pl.Trainer
    ):
        overlayedImages = []
        for img, label, pred  in zip(images, labels, preds):
            overlayedImages.append(wandb.Image(img, masks={
                "ground_truth": {
                    "mask_data": label,
                    "class_labels": class_labels
                },
                "predictions": {
                    "mask_data": pred,
                    "class_labels": class_labels
                },
            }))

        trainer.logger.experiment.log({f"validation/images_{modality}": overlayedImages}) # type: ignore
        overlayedImages = []
