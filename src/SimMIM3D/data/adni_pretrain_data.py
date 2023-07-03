import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    ToTensord,
)

from src.SimMIM3D.data.datasets import AdniDataset
from seg3d.src.mlutils.transforms import MaskGenerator3D

class AdniPretrainData(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            num_workers: int,
            img_size: int = 128,
            seed = 42,
            patch_size: int = 1,
            mask_ratio: float = 0.0
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = (img_size,) * 3
        self.seed = seed

        self.train_transform = Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image"]),
            NormalizeIntensityd(keys="image", channel_wise=True),
            RandSpatialCropd(keys=["image"], roi_size=(120, 120, 120), random_size=True),
            Resized(keys=["image"], spatial_size=(128, 128, 128)),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandAdjustContrastd(keys=["image"], prob=0.7, gamma=(0.5, 2.5)),
            RandShiftIntensityd(keys=["image"], offsets=0.125, prob=0.7),
            MaskGenerator3D(img_size=img_size, mask_ratio=mask_ratio, mask_patch_size=patch_size),
            ToTensord(keys=["image"], dtype=torch.float),
        ])

        self.val_transform = Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image"]),
            NormalizeIntensityd(keys="image", channel_wise=True),
            RandSpatialCropd(keys=["image"], roi_size=(120, 120, 120), random_size=True),
            Resized(keys=["image"], spatial_size=[128, 128, 128]),
            MaskGenerator3D(img_size=img_size, mask_ratio=mask_ratio, mask_patch_size=patch_size),
            ToTensord(keys=["image"], dtype=torch.float),
        ])

        self.test_transform = Compose([
            ToTensord(keys=["image"], dtype=torch.float)
        ])


    def setup(self, stage=None):
        if stage == 'fit':
            self.train_ds = AdniDataset(
                root_dir=self.data_dir, 
                transform=self.train_transform,
                section="training",
                seed=self.seed,
            )
        
            self.val_ds = AdniDataset(
                root_dir=self.data_dir, 
                transform=self.val_transform,
                section="validation",
                seed=self.seed,
            )

        if stage == 'test' or stage is None:
            self.test_ds = AdniDataset(
                root_dir=self.data_dir, 
                transform=self.test_transform,
                section="test",
                seed=self.seed,
            )
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
