import pytorch_lightning as pl
import torch

from monai.optimizers import WarmupCosineSchedule
from monai.losses import DiceLoss
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric # type: ignore
from monai.metrics.meandice import compute_dice

class FinetuneUNETR(pl.LightningModule):
    def __init__(
            self,
            net,
            learning_rate: float, 
            weight_decay: float,
            warmup_epochs: int,
            epochs: int,
            num_classes: int,
        ):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.num_classes = num_classes

        self.loss_fn = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
        )

        self.validation_step_outputs = []
        self.val_dice_score = DiceMetric(include_background=False, num_classes=self.num_classes, reduction="mean_batch")

        # Logging
        self.save_hyperparameters(ignore=["net", "learning_rate", "loss_fn"])


    def configure_optimizers(self):
        # TODO: Consider layer-wise learning rate decay (layer decay ratio = 0.75) to
        # stabilize the ViT training as suggested in "Self-Pretraining with MAE"
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )

        return {"optimizer": optimizer}
        # TODO: reenable warmup scheduler
        # lr_scheduler = WarmupCosineSchedule(
        #     optimizer=optimizer,
        #     warmup_steps=self.warmup_epochs,
        #     t_total=self.epochs + self.warmup_epochs
        # )
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}

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

        self.log("training/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss, "y_pred": y_pred}
    

    # Validation
    def validation_step(self, batch, batch_idx):
        loss, y_pred = self.common_step(batch, batch_idx) 
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)

        self.validation_step_outputs.append({"label": batch["label"], "y_pred": y_pred})
        self.log("validation/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss, "y_pred": y_pred, "label": batch['label']}

    def on_validation_epoch_end(self):
        label = torch.cat([output["label"].cpu() for output in self.validation_step_outputs])
        y_hat = torch.cat([output["y_pred"].cpu() for output in self.validation_step_outputs])

        # ET = 3            axis = 1
        # WT = 3 + 1 + 2    axis = 2
        # TC = 3 + 1        axis = 3

        dice_avg = 0

        et_label = one_hot(torch.where(label == 3, 1, 0), num_classes=2)
        et_y_hat = one_hot(torch.where(y_hat == 3, 1, 0), num_classes=2)
        dice = compute_dice(et_label, et_y_hat, include_background=False)
        dice = torch.nan_to_num(dice, nan=0)
        dice = torch.mean(dice, dim=0)
        self.log("validation/dice_et", dice, sync_dist=True)
        dice_avg += dice
        del et_label
        del et_y_hat

        wt_label = one_hot(torch.where((label == 3) | (label == 2) | (label == 1), 1, 0), num_classes=2)
        wt_y_hat = one_hot(torch.where((y_hat == 3) | (y_hat == 2) | (y_hat == 1), 1, 0), num_classes=2)
        dice = compute_dice(wt_label, wt_y_hat, include_background=False)
        dice = torch.nan_to_num(dice, nan=0)
        dice = torch.mean(dice, dim=0)
        self.log("validation/dice_wt", dice, sync_dist=True)
        dice_avg += dice
        del wt_label
        del wt_y_hat

        tc_label = one_hot(torch.where((label == 2) | (label == 3), 1, 0), num_classes=2)
        tc_y_hat = one_hot(torch.where((y_hat == 3) | (y_hat == 1), 1, 0), num_classes=2)
        dice = compute_dice(tc_label, tc_y_hat, include_background=False)
        dice = torch.nan_to_num(dice, nan=0)
        dice = torch.mean(dice, dim=0)
        self.log("validation/dice_tc", dice)
        dice_avg += dice
        del tc_label
        del tc_y_hat

        dice_avg /= 3
        self.log("validation/dice", dice_avg, sync_dist=True)
        self.validation_step_outputs.clear()
    

    # Testing
    def test_step(self, batch, batch_idx):
        loss, y_pred = self.common_step(batch, batch_idx) 
        return {"loss": loss, "y_pred": y_pred}
