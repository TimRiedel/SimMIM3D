import torch
from monai.transforms import Transform, MapTransform

class MapToSequentialChannels(Transform):
    """
    Labels in the BraTS dataset are stored in channels [0, 1, 2, 4].
    This transform converts them to the sequential order [0, 1, 2, 3].
    """
    def __call__(self, data):
        data[data==4] = 3
        return data


class MapToSequentialChannelsd(MapTransform):
    """
    Labels in the BraTS dataset are stored in channels [0, 1, 2, 4].
    Channel 0 does not store any relevant data, only empty space, so it can be discarded.
    This transform converts them to the sequential order [1, 2, 3].
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            result.append(d[key] == 3)
            d[key] = torch.stack(result, axis=0)
        return d