import numpy as np
import wandb
import pytorch_lightning as pl


def log_images(
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    modality: str,
    class_labels: dict,
    trainer: pl.Trainer
):
    overlayed_images = []
    for img, label, pred in zip(images, labels, preds):
        overlayed_images.append(wandb.Image(img, masks={
            "ground_truth": {
                "mask_data": label,
                "class_labels": class_labels
            },
            "predictions": {
                "mask_data": pred,
                "class_labels": class_labels
            },
        }))

    trainer.logger.experiment.log({f"validation/images_{modality}": overlayed_images})  # type: ignore
    overlayed_images = []


class LogBratsValidationPredictions(pl.Callback):
    def __init__(
        self,
        num_samples,
        class_labels=None
    ):
        super().__init__()
        if class_labels is None:
            class_labels = {1: "ED", 2: "NET", 3: "ET"}
        self.num_samples = num_samples
        self.class_labels = class_labels

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
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
        log_images(images, labels, preds, modality="t2_t1gd", class_labels=self.class_labels, trainer=trainer)
