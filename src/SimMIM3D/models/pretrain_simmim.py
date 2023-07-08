import pytorch_lightning as pl
import wandb
import torch
import torch.nn.functional as F
from monai.optimizers import WarmupCosineSchedule

class PretrainSimMIM(pl.LightningModule):
    def __init__(
            self,
            net,
            learning_rate: float, 
            weight_decay: float,
            warmup_epochs: int,
            epochs: int
        ):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs

        # Logging
        self.save_hyperparameters(ignore=["net", "learning_rate"])
        self.num_samples = 10
        self.channel_idx = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
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
        return batch["image"], batch["mask"]

    def common_step(self, x, mask):
        x_rec = self.net(x, mask) 

        # get masked representation of input
        patch_size = self.net.encoder.patch_size
        mask = mask.repeat_interleave(patch_size[0], 1).repeat_interleave(patch_size[1], 2).repeat_interleave(patch_size[2], 3).unsqueeze(1).contiguous() # type: ignore
        x_masked = x * (1 - mask)

        # calculate loss on masked patches only
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.net.encoder.in_channels
        return loss, x_rec, x_masked


    # Training
    def training_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_rec, _ = self.common_step(x, mask)
        return {"loss": loss, "x_rec": x_rec}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("training/loss", outputs["loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["image"].shape[0]) # type: ignore
    

    # Validation
    def validation_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_rec, x_masked = self.common_step(x, mask)
        return {"loss": loss, "x_rec": x_rec, "x_masked": x_masked}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log("validation/loss", outputs["loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["image"].shape[0])

        if batch_idx == 0:
            slice_idx = batch["image"].shape[4] // 2

            if self.current_epoch == 0:
                images = batch["image"][:self.num_samples, self.channel_idx, :, :, slice_idx].detach().cpu()
                self.logger.experiment.log({"validation/original": [wandb.Image(img) for img in images]}) # type: ignore

            masked = outputs["x_masked"][:self.num_samples, self.channel_idx, :, :, slice_idx].detach().cpu()
            reconstructions = outputs["x_rec"][:self.num_samples, self.channel_idx, :, :, slice_idx].detach().cpu()
            self.logger.experiment.log({"validation/masked": [wandb.Image(mask) for mask in masked]}) # type: ignore
            self.logger.experiment.log({"validation/reconstruction": [wandb.Image(recon) for recon in reconstructions]}) # type: ignore


    # Testing
    def test_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        loss, x_rec, x_masked = self.common_step(x, mask)
        return {"loss": loss, "x_rec": x_rec, "x_masked": x_masked}

    
    # Prediction
    def predict_step(self, batch, batch_idx):
        x, mask = self.prepare_batch(batch)
        x_pred = self.net(x, mask)
        return x_pred

