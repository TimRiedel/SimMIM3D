import torch
import numpy as np
from monai.transforms.transform import MapTransform

class ConvertToBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats17 classes:
    label 1 is the edema
    label 2 is the non-enhancing tumor
    label 3 is the enhancing tumor

    Generated classes are:
    WT (Whole Tumor): labels 1 (edema) and 2 (tumor non-enh) and 3 (tumor enh)
    TC (Tumor Core): labels 2 (tumor non-enh) and 3 (tumor enh)
    ET (Enhancing Tumor): label 3 (tumor enh)

    Based on MONAI's `ConvertToMultiChannelBasedOnBratsClassesd`.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]

            if label.ndim == 4 and label.shape[0] == 1:
                label = label.squeeze(0)

            result = [(label == 1) | (label == 2) | (label == 3), (label == 2) | (label == 3), (label == 3)]
            d[key] = torch.stack(result, dim=0) if isinstance(label, torch.Tensor) else np.stack(result, axis=0)
        return d