import torch
from time import time
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
from src.mlutils.transforms import MaskGenerator3D

data_dir = "/dhc/home/tim.riedel/bachelor-thesis/data/ADNI"
img_size = 128
mask_ratio = 0.7
patch_size = 16

train_transform = Compose([
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

train_ds = AdniDataset(
    root_dir=data_dir, 
    transform=train_transform,
    train_frac=0.1,
    val_frac=0.1,
    section="training",
    is_pretrain=False,
    seed=42,
)

for batch_size in (8, 16, 32):
    print("-----------------------------------------")
    print(f"Batch size: {batch_size}")
    for num_workers in range(2, 32, 2):  
        train_loader = DataLoader(train_ds, shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
        start = time()
        for i, data in enumerate(train_loader, 0):
            pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
    print()