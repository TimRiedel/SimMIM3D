# my_project/config.py
import os
from yacs.config import CfgNode as CN
import pathlib

HOME_DIR = os.getcwd()

_C = CN()

_C.SYSTEM = CN()
# Accelerator to use in the experiment
_C.SYSTEM.ACCELERATOR = "gpu"
# Strategy for single / multiple GPUs
_C.SYSTEM.STRATEGY = "ddp"
# Number of GPUS to use in the experiment
_C.SYSTEM.DEVICES = -1
# Number of nodes on a distributed system
_C.SYSTEM.NODES = 1
# Floating point precision for weights
_C.SYSTEM.PRECISION = "16-mixed"


_C.LOGGING = CN()
# Version of the model
_C.LOGGING.VERSION = "1"
# Name of the run
_C.LOGGING.RUN_NAME = "SimMIM3D_PT"
# Directory for jobs (checkpoints, logs, etc.)
_C.LOGGING.JOBS_DIR = f"{HOME_DIR}/jobs"


_C.DATA = CN()
# Directory for BraTS2017 dataset
_C.DATA.BRATS_DATA_DIR = f"{HOME_DIR}/data/BraTS2017"
# Input image size in one dimension
_C.DATA.IMG_SIZE = 96
# Batch size for a single GPU
_C.DATA.BATCH_SIZE = 8
# Number of data loading threads
_C.DATA.NUM_WORKERS = 16
# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_RATIO = 0.75


_C.TRAINING = CN()
# Learning rate for the optimizer
_C.TRAINING.BASE_LR = 3e-4
# Weight decay for the optimizer
_C.TRAINING.WEIGHT_DECAY = 0.01
# Number of warmup epochs for learning rate scheduler
_C.TRAINING.WARMUP_EPOCHS = 100
# Number of epochs to train for
_C.TRAINING.EPOCHS = 900
# Number of cross validations / folds to run
_C.TRAINING.CROSS_VALIDATIONS = 1


_C.MODEL = CN()
# Number of input channels
_C.MODEL.IN_CHANNELS = 4
# Size of one patch
_C.MODEL.PATCH_SIZE = 16
# Dropout rate for the encoder
_C.MODEL.ENCODER_DROPOUT = 0.0


def get_config(args = None):
    """Get a yacs CfgNode object with values for pre-training or fine-tuning."""
    path = pathlib.Path(__file__).parent.resolve()

    if args.finetune:
        _C.merge_from_file(f"{str(path)}/configs/simmim_vit_finetune.yaml")
    else:
        _C.merge_from_file(f"{str(path)}/configs/simmim_vit_pretrain.yaml")
    
    return _C.clone()


# Source: https://github.com/rbgirshick/yacs/issues/19
def convert_cfg_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """

    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict