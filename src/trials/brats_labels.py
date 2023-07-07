
import random
import numpy as np
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    EnsureTyped, 
    LoadImaged, 
    Orientationd,
)
import torch
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

max_size = (192, 192, 128)
input_size = (128, 128, 128)

train_transform = Compose([
    LoadImaged(keys=["image"], image_only=True),
    LoadImaged(keys=["label"], dtype=np.uint8, image_only=True),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    SpatialCropd(keys=["image", "label"], roi_size=max_size, roi_center=(120, 120, 81)),
    RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    NormalizeIntensityd(keys="image", channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
])


train_ds = DecathlonDataset(
    root_dir="/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017", 
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    num_workers=1,
    cache_rate=0.0,
)

ds = train_ds
idx = random.randint(0, len(ds))
data = train_ds[idx]
img, label = data["image"], data["label"] # type: ignore
print(label.shape)
print(np.unique(label))