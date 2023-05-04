import sys
import random
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
from monai.utils import set_determinism, first
from monai.data import ArrayDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    CastToType,
    SpatialCrop,
    AsDiscrete
)

from src.mlutils.transforms import MapToSequentialChannels, CombineBraTSChannels
from src.mlutils.plotting import plot_image, plot_segmentation

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

set_determinism(seed=0)

# Command Line Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n_images", help="Number of images to be processed.", action="store")
parser.add_argument("--validation", help="Import the validation dataset. By default the training dataset is imported.", action="store_true")
args = parser.parse_args()

set_type = "training"
if (args.validation):
    print("Importing validation dataset.")
    set_type = "validation"


DATA_PATH = "/dhc/home/tim.riedel/bachelor-thesis/data"
BRATS_DATASET_PATH = f"{DATA_PATH}/BraTS2020/BraTS2020_{set_type.capitalize()}Data"
OUTPUT_PATH = f"{DATA_PATH}/BraTS2020_Processed/{set_type}"


# Get Image and Segmentation Paths
flair_paths = sorted(glob.glob(f"{BRATS_DATASET_PATH}/*/*_flair.nii"))
t1_paths = sorted(glob.glob(f"{BRATS_DATASET_PATH}/*/*_t1.nii"))
t1ce_paths = sorted(glob.glob(f"{BRATS_DATASET_PATH}/*/*_t1ce.nii"))
t2_paths = sorted(glob.glob(f"{BRATS_DATASET_PATH}/*/*_t2.nii"))
seg_paths = sorted(glob.glob(f"{BRATS_DATASET_PATH}/*/*_seg.nii"))


# Combine all channels for each image
imgs = []
for i in range(len(flair_paths)):
    imgs.append({ "flair": flair_paths[i], "t1": t1_paths[i], "t1ce": t1ce_paths[i], "t2": t2_paths[i] })

random.seed(a=3)
if (args.n_images):
    imgs = random.sample(imgs, int(args.n_images))


# Define transforms
img_transform = Compose(
    [
        CombineBraTSChannels(),
        EnsureChannelFirst(channel_dim=0),
        Orientation(axcodes="RAS"),
        SpatialCrop(roi_size=(192, 192, 128), roi_center=(120, 125, 81)),
    ]
)

seg_transform = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        Orientation(axcodes="RAS"),
        # CastToType(dtype=np.uint8),
        MapToSequentialChannels(),
        SpatialCrop(roi_size=(192, 192, 128), roi_center=(120, 125, 81)),
    ]
)


# Create dataset
img_dataset = ArrayDataset(imgs, img_transform, seg_paths, seg_transform)
img_loader = DataLoader(img_dataset, batch_size=1, num_workers=4, shuffle=False)


# Iterate over dataset and store results
idx = 1;
for img_batch, seg_batch in img_loader:
    img = img_batch[0]
    seg = seg_batch[0]
    print("------------------------------------")
    print(f"Preparing next image and mask: {idx}")

    # # Only use data with at least 1% useful annotated volume
    # val, counts = np.unique(seg, return_counts=True)
    # percentAnnotated = 1 - counts[0] / sum(counts, start=1)
    # if percentAnnotated < 0.01:
    #     print("Skipping image")
    #     continue

    print("Saving image")
    # Convert to categorical segmentation
    seg = AsDiscrete(to_onehot=4)(seg)

    # plot_image(img)
    # plot_segmentation(seg)
    
    np.save(f"{OUTPUT_PATH}/images/image_{idx}.npy", img)
    np.save(f"{OUTPUT_PATH}/masks/mask_{idx}.npy", seg)
    idx += 1