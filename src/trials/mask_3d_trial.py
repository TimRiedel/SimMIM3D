from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.networks.blocks import PatchEmbeddingBlock
from monai.transforms import (
    EnsureTyped,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from seg3d.src.mlutils.transforms import MaskGenerator3D

DATA_DIR = '/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017'
input_size = 96
img_size = (input_size, input_size, input_size)
in_chans = 4
embed_dim = 768
patch_size = 16
model_patch_size = 16

transform = Compose([
    LoadImaged(keys='image', ensure_channel_first=True),
    EnsureTyped(keys="image"),
    Orientationd(keys=["image"], axcodes="RAS"),
    RandSpatialCropd(keys=["image"], roi_size=img_size, random_size=False),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    NormalizeIntensityd(keys="image", channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    MaskGenerator3D(),
])

dataset = DecathlonDataset(
    root_dir=DATA_DIR,
    task='Task01_BrainTumour',
    section='training',
    transform=transform,
    download=False,
    cache_rate=0.0,
)

dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

data = next(iter(dataloader))
original = data['image']
mask = data['mask']


# Constants
patch_embedding = PatchEmbeddingBlock(
    in_channels=in_chans,
    img_size=img_size,
    patch_size=patch_size,
    hidden_size=embed_dim,
    num_heads=12,
    pos_embed="conv"
)

mask_ratio = 0.75
mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


# Patch Embedding
x = original
print(f"PATCH: x shape: {x.shape}")
x = patch_embedding(x) 
print(f"PATCH: x shape after embedding: {x.shape}")


# Apply Mask
# mask = torch.tensor(mask, dtype=torch.float32)
B, L, E = x.shape
print(f"APPLY: of x is B: {B}, L: {L}, E: {E}")
print(f"APPLY: mask_token shape before expansion: {mask_token.shape}")
mask_token = mask_token.expand(B, L, -1)
print(f"APPLY: mask_token shape after expansion: {mask_token.shape}")
print(f"APPLY: mask shape before expansion: {mask.shape}")
w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
print(f"APPLY: mask shape flattening shape: {w.shape}")
x = x * (1 - w) + mask_token * w
print(f"APPLY: x shape: {x.shape}")
print("\n")

# Reshape
B, L, E = x.shape
print(f"RESHAPE: of x is B: {B}, L: {L}, C: {E}")
H = W = D = round(L ** (1./3.))
print(f"RESHAPE: H: {H}, W: {W}, D: {D}")
x = x.permute(0, 2, 1)
print(f"RESHAPE: x shape after permute: {x.shape}")
x = x.reshape(B, E, H, W, D)
print(f"RESHAPE: x shape after reshape: {x.shape}")
print("\n")

# Decode
decoder = nn.ConvTranspose3d(
    in_channels=embed_dim,
    out_channels=in_chans,
    kernel_size=patch_size,
    stride=patch_size,
    bias=False,
)
# decoder.weight.data.fill_(1)

print(f"RECON: x shape: {x.shape}")
x_rec = decoder(x)
print(f"RECON: x_rec shape: {x_rec.shape}")

# Display
original_image = original[0].detach().numpy()
reconstructed_image = x_rec[0].detach().numpy()
fig, axs = plt.subplots(2, 4)


axs[0, 0].imshow(original_image[0, :, :, 80], cmap='gray')
axs[0, 0].set_title('Orig. C0')
axs[0, 1].imshow(original_image[1, :, :, 80], cmap='gray')
axs[0, 1].set_title('Orig. C1')
axs[0, 2].imshow(original_image[2, :, :, 80], cmap='gray')
axs[0, 2].set_title('Orig. C2')
axs[0, 3].imshow(original_image[3, :, :, 80], cmap='gray')
axs[0, 3].set_title('Orig. C3')

axs[1, 0].imshow(reconstructed_image[0, :, :, 80], cmap='gray')
axs[1, 0].set_title('Recon. C0')
axs[1, 1].imshow(reconstructed_image[1, :, :, 80], cmap='gray')
axs[1, 1].set_title('Recon. C1')
axs[1, 2].imshow(reconstructed_image[2, :, :, 80], cmap='gray')
axs[1, 2].set_title('Recon. C2')
axs[1, 3].imshow(reconstructed_image[3, :, :, 80], cmap='gray')
axs[1, 3].set_title('Recon. C3')

print(f"RECON: Original [0, 0, 0, 80]: {original_image[0, 0, 0, 80]}")
print(f"RECON: Original [0, 6, 0, 80]: {original_image[1, 6, 0, 80]}")
print(f"RECON: Original [0, 12, 0, 80]: {original_image[2, 12, 0, 80]}")
print(f"RECON: Original [0, 18, 0, 80]: {original_image[3, 18, 0, 80]}")

print(f"RECON: Reconstructed [0, 0, 0, 80]: {reconstructed_image[0, 0, 0, 80]}")
print(f"RECON: Reconstructed [0, 6, 0, 80]: {reconstructed_image[1, 6, 0, 80]}")
print(f"RECON: Reconstructed [0, 12, 0, 80]: {reconstructed_image[2, 12, 0, 80]}")
print(f"RECON: Reconstructed [0, 18, 0, 80]: {reconstructed_image[3, 18, 0, 80]}")

# show the plot
plt.show()
plt.savefig('masking3d.png')