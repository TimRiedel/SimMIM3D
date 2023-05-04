from monai.transforms import Transform

class MapToSequentialChannels(Transform):
    """
    Labels in the BraTS dataset are stored in channels [0, 1, 2, 4].
    This transform converts them to the sequential order [0, 1, 2, 3].
    """
    def __call__(self, data):
        data[data==4] = 3
        return data