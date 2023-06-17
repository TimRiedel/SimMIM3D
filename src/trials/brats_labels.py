
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

from src.mlutils.plotting import plot_segmentation_file
from src.mlutils.transforms import ConvertToBratsClassesd

max_size = (192, 192, 128)
input_size = (128, 128, 128)

train_transform = Compose([
    LoadImaged(keys=["image"]),
    LoadImaged(keys=["label"], dtype=np.uint8),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image", "label"]),
    ConvertToBratsClassesd(keys="label"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
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
plot_segmentation_file(label)