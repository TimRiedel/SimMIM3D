
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
from monai.data import NibabelWriter
from monai.transforms import (
    BorderPadd,
    CenterSpatialCropd,
    CropForegroundd,
    Compose, 
    EnsureChannelFirstd, 
    EnsureTyped, 
    LoadImaged, 
    NormalizeIntensityd,
    Orientationd,
    SpatialCropd, 
    SpatialPadd,
    Resized,
    RandFlipd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ToTensord,
)
import torch

from src.mlutils.plotting.plot import plot_image_file

max_size = (170, 170, 170)
input_size = (128, 128, 128)

train_transform = Compose([
    LoadImaged(keys=["image"], image_only=True),
    LoadImaged(keys=["label"], image_only=True, dtype=np.uint8),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    # Constant Resizing and Normalization
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=max_size, mode="constant", constant_values=0),
    Resized(keys=["image"], spatial_size=input_size, mode="area"),
    Resized(keys=["label"], spatial_size=input_size, mode="nearest"),
    NormalizeIntensityd(keys=["image"], channel_wise=True),
    # Rand Resizing
    RandSpatialCropd(keys=["image", "label"], roi_size=(120, 120, 120), random_size=True),
    Resized(keys=["image"], spatial_size=input_size, mode="area"),
    Resized(keys=["label"], spatial_size=input_size, mode="nearest"),
    # Augmentation
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandAdjustContrastd(keys=["image"], gamma=(0.5, 1.5), prob=0.5),
    RandScaleIntensityd(keys=["image"], factors=0.2, prob=1.0),
    RandShiftIntensityd(keys=["image"], offsets=0.2, prob=1.0),
    ToTensord(keys=["image"], dtype=torch.float)
])


train_ds = DecathlonDataset(
    root_dir="/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017", 
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    num_workers=4,
    cache_rate=0.0,
)

ds = train_ds
# idx = random.randint(0, len(ds))
# data = train_ds[idx]
# img, label = data["image"], data["label"] # type: ignore
# plot_image_file(img)
# print(img.shape)
# print(label.shape)
# print(np.unique(label))

for i in range(10):
    idx = random.randint(0, len(ds))
    data = train_ds[idx]
    img, label = data["image"], data["label"] # type: ignore
    plot_image_file(img, f"brats-img-{i}.png")
# writer = NibabelWriter()
# writer.set_data_array(data["image"])
# writer.write("/dhc/home/tim.riedel/bachelor-thesis/brats-nopad.nii.gz")