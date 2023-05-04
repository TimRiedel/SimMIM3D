import numpy as np
from monai.transforms import Compose, LoadImage, ScaleIntensity
from monai.data import Dataset 
from monai.transforms import Transform, LoadImage, Orientation, ScaleIntensity

class CombineBraTSChannels(Transform):
    """
    Combines the four BraTS channels into a single image, which are flair, t1, t1ce and t2.
    The data must be in the form of a dictionary with the keys "flair", "t1", "t1ce" and "t2".
    It returns a 4D numpy array with all channels stacked.
    """
    def __call__(self, data):
        img_paths = [*data.values()];
        img_transform = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
            ]
        )

        img_dataset = Dataset(img_paths, img_transform)
        images = img_dataset[:]
        image_stack = np.stack([*images], axis=0)
        return image_stack