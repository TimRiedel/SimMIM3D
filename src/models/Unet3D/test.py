import random
from matplotlib import pyplot as plt
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
    RandShiftIntensityd, 
    RandSpatialCropd,
)

from src.mlutils.plotting import plot_img_seg_file, overlay_segmentation_image
from src.mlutils.transforms import MapToSequentialChannelsd

train_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image", "label"]),
    MapToSequentialChannelsd(keys="label"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    SpatialCropd(keys=["image", "label"], roi_size=(192, 192, 128), roi_center=(120, 120, 81)),
    RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
])

val_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image", "label"]),
    MapToSequentialChannelsd(keys="label"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    SpatialCropd(keys=["image", "label"], roi_size=(192, 192, 128), roi_center=(120, 120, 81)),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])

train_ds = DecathlonDataset(
    root_dir="/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017", 
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    num_workers=4,
    cache_rate=0.0,
)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

val_ds = DecathlonDataset(
    root_dir="/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017",
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    num_workers=4,
    cache_rate=0.0
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

test_ds = DecathlonDataset(
    root_dir="/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017",
    task="Task01_BrainTumour",
    transform=val_transform,
    section="test",
    num_workers=4,
    cache_rate=0.0
)

# Plot some images
ds = train_ds
idx = random.randint(0, len(ds))
ds_data = ds[idx]
print(f"Length of Dataset: {len(ds)}")
print(f"Index: {idx}")
print(f"Image Shape: {ds_data['image'].shape}")
print(f"Label Shape: {ds_data['label'].shape}")

fig = overlay_segmentation_image(
    img=ds_data["image"], 
    truth_label=ds_data["label"], 
    pred_label=ds[idx+1]["label"], 
    label_channel=1, 
    slice_no=81, 
    save_file=True, 
    img_channel=0
)

print(fig)