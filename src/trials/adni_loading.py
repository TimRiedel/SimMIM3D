import random
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    ToTensord,
)

from src.SimMIM3D.data.datasets import AdniDataset
from src.mlutils.plotting import plot_image_file

ADNI_DATA_PATH = "/dhc/home/tim.riedel/bachelor-thesis/data/ADNI"

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
    ToTensord(keys=["image"], dtype=torch.float),
])

train_ds = AdniDataset(
    root_dir=ADNI_DATA_PATH, 
    transform=train_transform,
    section="training",
    seed=42,
)

idx = random.randint(0, len(train_ds))
data = train_ds[idx]
# print(data)
print(data["image"].shape)
print(data["label"])
print(data["diagnosis"])

plot_image_file(data["image"])