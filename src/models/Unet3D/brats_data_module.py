import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from monai.apps import DecathlonDataset
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet, BasicUNet
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    EnsureTyped, 
    LoadImaged, 
    NormalizeIntensityd,
    Orientationd,
    SpatialCropd, 
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd, 
    RandSpatialCropd,
    ConvertToMultiChannelBasedOnBratsClassesd
)

from src.mlutils.transforms import MapToSequentialChannelsd

class BratsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, input_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

        max_size = torch.tensor((192, 192, 128))
        assert torch.all(torch.tensor(input_size) <= max_size), "Not all dimensions of `input_size` are less than or equal to `(192, 192, 128)`"

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
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
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
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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
                section="validation",
                num_workers=self.num_workers,
                cache_rate=0.0,
            )
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)