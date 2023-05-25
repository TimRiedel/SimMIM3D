from src.models.MAE import SimMIM3D
from torchinfo import summary

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

network = SimMIM3D(img_size=(96, 96, 96))
summary(network, (1, 4, 96, 96, 96))
