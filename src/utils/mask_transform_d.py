import numpy as np
import torch
from monai.transforms import Transform


class MaskGenerator3D(Transform):
    def __init__(self, img_size: int = 96, mask_patch_size: int = 16, mask_ratio: float = 0.75):
        self.input_size = img_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = mask_patch_size
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

        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)

        d = dict(data)
        d['mask'] = torch.tensor(mask, dtype=torch.float32)
        return d
