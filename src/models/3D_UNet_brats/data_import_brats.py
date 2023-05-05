#%%
import glob
import torch

from torchinfo import summary
import pytorch_lightning as pl
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet, BasicUNet
from monai.transforms import Compose, LoadImaged, RandFlipd, RandScaleIntensityd, RandShiftIntensityd

from src.mlutils.plotting import plot_image, plot_segmentation
from .model import UNet3D

DATA_PATH = "/dhc/home/tim.riedel/bachelor-thesis/data"
BRATS_TRAIN_DATASET_PATH = f"{DATA_PATH}/BraTS2020_Processed/training/"
BRATS_VAL_DATASET_PATH = f"{DATA_PATH}/BraTS2020_Processed/validation/"

train_img_paths = sorted(glob.glob(f"{BRATS_TRAIN_DATASET_PATH}/images/*"))
train_seg_paths = sorted(glob.glob(f"{BRATS_TRAIN_DATASET_PATH}/masks/*"))
val_img_paths = sorted(glob.glob(f"{BRATS_VAL_DATASET_PATH}/images/*"))
val_seg_paths = sorted(glob.glob(f"{BRATS_VAL_DATASET_PATH}/masks/*"))

train_dict = []
for i in range(len(train_img_paths)):
    train_dict.append({ "image": train_img_paths[i], "seg": train_seg_paths[i]}) 
val_dict = []
for i in range(len(val_img_paths)):
    val_dict.append({ "image": val_img_paths[i], "seg": val_seg_paths[i]}) 

# Define transforms
train_transform = Compose(
    [
        LoadImaged(keys=["image", "seg"]),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "seg"]),
    ]
)

# Create dataset
train_ds = Dataset(data=train_dict, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
val_ds = Dataset(data=val_dict, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
#%%

network = UNet(
    spatial_dims=3,
    in_channels=4, # 4 modalities (FLAIR, T1, T1ce, T2)
    out_channels=4, # 4 segmentation classes
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2, #TODO: Understand
    dropout=0.1, #TODO: Understand
)
summary(network, input_size=(1, 4, 192, 192, 128))

model = UNet3D(
    net=network,
    num_classes=4,
    loss_fn=None, #TODO: Choose loss function
    learning_rate=1e-2, #TODO: Choose learning rate
    optimizer_class=torch.optim.AdamW, #TODO: Choose optimizer
)

trainer = pl.Trainer(accelerator="gpu")
trainer.fit(model, train_loader, val_loader)
