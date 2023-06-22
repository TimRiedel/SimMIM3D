import numpy as np
import torch
import wandb
import pytorch_lightning as pl

class LogBratsValidationPredictions(pl.Callback):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):
        """Logs validation images and predictions for the first batch and
        the given channel index. Assumes outputs has a field 'y_pred'."""

        if batch_idx != 0:
            return

        slice_idx = batch["image"].shape[4] // 2

        images = batch["image"][:self.num_samples, :, :, :, slice_idx].detach().cpu()
        one_hot_labels = batch["label"][:self.num_samples, :, :, :, slice_idx].detach().cpu().numpy()
        one_hot_preds = outputs["y_pred"][:self.num_samples, :, :, :, slice_idx].detach().cpu().numpy()

        # WT (label 0) is best visible on T2 (channel 3)
        t2_images = images[:, 3]
        wt_labels = one_hot_labels[:, 0]
        wt_preds = one_hot_preds[:, 0]
        self.log_images(t2_images, wt_labels, wt_preds, "t2", {1: "WT"}, trainer)

        # ET (label 2) is best visible on T1gd (channel 2)
        t1gd_images = images[:, 2]
        et_labels = one_hot_labels[:, 2]
        et_preds = one_hot_preds[:, 2]
        self.log_images(t1gd_images, et_labels, et_preds, "t1gd", {1: "ET"}, trainer)

        # TC (label 1) is best visible on a combination of T2 (channel 3) and T1gd (channel 2)
        t2_t1gd_images = t2_images + t1gd_images
        tc_labels = one_hot_labels[:, 1]
        tc_preds = one_hot_preds[:, 1]
        self.log_images(t2_t1gd_images, tc_labels, tc_preds, "t2_t1gd", {1: "TC"}, trainer)


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

        trainer.logger.experiment.log({f"validation/images_{modality}_{class_labels[1]}": overlayedImages}) # type: ignore
        overlayedImages = []
