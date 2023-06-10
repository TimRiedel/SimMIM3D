import numpy as np
import pytorch_lightning as pl
import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose, 
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd, 
    EnsureTyped, 
    LoadImaged, 
    NormalizeIntensityd,
    Orientationd,
    SpatialCropd, 
    RandFlipd,
    RandScaleIntensityd,
    RandSpatialCropd,
)

class BratsData(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            num_workers: int,
            img_size: int = 96,
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = (img_size,) * 3

        max_size = (192, 192, 128)
        assert torch.all(torch.tensor(self.input_size) <= torch.tensor(max_size)), "Not all dimensions of `input_size` are less than or equal to `(192, 192, 128)`"

        self.train_transform = Compose([
            LoadImaged(keys=["image"]),
            LoadImaged(keys=["label"], dtype=np.uint8),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            SpatialCropd(keys=["image", "label"], roi_size=max_size, roi_center=(120, 120, 81)),
            RandSpatialCropd(keys=["image", "label"], roi_size=self.input_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        ])

        self.val_transform = Compose([
            LoadImaged(keys=["image"]),
            LoadImaged(keys=["label"], dtype=np.uint8),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            SpatialCropd(keys=["image", "label"], roi_size=max_size, roi_center=(120, 120, 81)),
            RandSpatialCropd(keys=["image", "label"], roi_size=self.input_size, random_size=False),
            NormalizeIntensityd(keys="image", channel_wise=True),
        ])


    def setup(self, stage=None):
        if stage == 'fit':
            self.train_ds = DecathlonDataset(
                root_dir=self.data_dir, 
                task="Task01_BrainTumour",
                transform=self.train_transform,
                section="training",
                num_workers=self.num_workers,
                cache_rate=0.0,
            )
        
            self.val_ds = DecathlonDataset(
                root_dir=self.data_dir,
                task="Task01_BrainTumour",
                transform=self.val_transform,
                section="validation",
                num_workers=self.num_workers,
                cache_rate=0.0,
            )

        if stage == 'test' or stage is None:
            self.test_ds = DecathlonDataset(
                root_dir=self.data_dir,
                task="Task01_BrainTumour",
                transform=self.val_transform,
                section="test",
                num_workers=self.num_workers,
                cache_rate=0.0,
            )
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)