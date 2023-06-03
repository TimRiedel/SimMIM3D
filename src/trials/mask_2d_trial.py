import numpy as np
import torch
import torch.nn as nn

input_size = 192
embed_dim = 768
batch_size = 1
patch_size = 32
model_patch_size = 32

mask_ratio = 0.75

rand_size = input_size // patch_size
scale = patch_size // model_patch_size
token_count = rand_size ** 2
mask_count = int(np.ceil(token_count * mask_ratio))

# Generation
mask_idx = np.random.permutation(token_count)[:mask_count]
mask = np.zeros(token_count, dtype=int)
mask[mask_idx] = 1

mask = mask.reshape((rand_size, rand_size))
mask = mask.repeat(scale, axis=0).repeat(scale, axis=1)
print(f"mask shape: {mask.shape}")
print(f"mask: {mask}")

# Add batch dimension
mask = torch.from_numpy(mask).unsqueeze(0)
print(f"mask shape after unsqueeze: {mask.shape}")




x = torch.randn(batch_size, token_count, embed_dim)
print(f"x shape: {x.shape}")

mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
print(f"mask_token shape before expansion: {mask_token.shape}")
mask_token = mask_token.expand(batch_size, token_count, -1)
print(f"mask_token shape after expansion: {mask_token.shape}")
w = mask.flatten(1)
print(f"w shap eafter flatten: {w.shape}")
w = w.unsqueeze(-1)
print(f"w shape after unsqueeze: {w.shape}")
w = w.type_as(mask_token)
print(f"w shape after type_as: {w.shape}")
x = x * (1 - w) + mask_token * w
print(f"x shape after masking: {x.shape}")