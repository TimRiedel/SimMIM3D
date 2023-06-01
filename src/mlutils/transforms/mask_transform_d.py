import numpy as np
import torch
from monai.transforms import MapTransform

class MaskGenerator3D(MapTransform):
    def __init__(self, input_size=96, input_channels=4, mask_patch_size=16, model_patch_size=1, mask_ratio=0.755555):
        self.input_size = input_size
        self.input_channels = input_channels
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 3
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self, data):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        d = dict(data)
        d['mask'] = torch.tensor(mask, dtype=torch.float32)
        return d