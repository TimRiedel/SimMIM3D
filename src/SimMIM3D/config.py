# type: ignore

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
_C.SYSTEM.PRECISION = "32-true"


_C.LOGGING = CN()
# Name of the wandb project
_C.LOGGING.PROJECT_NAME = "ba-thesis"
# Name of the run
_C.LOGGING.RUN_NAME = "Run"
# Directory for jobs (checkpoints, logs, etc.)
_C.LOGGING.JOBS_DIR = f"{HOME_DIR}/jobs"
# Directory to load checkpoint from
_C.LOGGING.CKPT_DIR = f"{HOME_DIR}/jobs/checkpoints"


_C.DATA = CN()
# Directory for  dataset
_C.DATA.DATA_DIR = ""
# Input image size in one dimension
_C.DATA.IMG_SIZE = 128
# Number of classes for multiclass segmentation
_C.DATA.NUM_CLASSES = 4
# Batch size for a single GPU
_C.DATA.BATCH_SIZE = 8
# Number of data loading threads
_C.DATA.NUM_WORKERS = 16
# [Pretraining] Mask patch size for MaskGenerator
_C.DATA.MASK_RATIO = 0.7
# [Finetuning] Percentage of original training set to use 
_C.DATA.TRAIN_FRAC = 1.0


_C.TRAINING = CN()
# Learning rate for the optimizer
_C.TRAINING.BASE_LR = 3e-4
# Weight decay for the optimizer
_C.TRAINING.WEIGHT_DECAY = 0.01
# [Finetuning] Number of warmup epochs in which encoder is frozen - set to 0 for no freezing or to EPOCHS for full freezing
_C.TRAINING.FREEZE_WARMUP_EPOCHS = 20
# Number of warmup epochs for learning rate scheduler
_C.TRAINING.LR_WARMUP_EPOCHS = 100
# Number of epochs to train for
_C.TRAINING.EPOCHS = 450
# Number of cross validation folds to run
_C.TRAINING.CV_FOLDS = 1


_C.MODEL = CN()
# Number of input channels
_C.MODEL.IN_CHANNELS = 4
# Size of one patch
_C.MODEL.PATCH_SIZE = 16
# Dropout rate for the encoder
_C.MODEL.ENCODER_DROPOUT = 0.0
# Path to checkpoint relative to checkpoint directory for finetuning, if empty no checkpoint is loaded
_C.MODEL.ENCODER_CKPT_PATH = "" 


def get_config(args = None):
    """Get a yacs CfgNode object with values for pre-training or fine-tuning."""
    path = pathlib.Path(__file__).parent.resolve()

    if args.finetune:
        _C.merge_from_file(f"{str(path)}/configs/unetr_finetune_{args.dataset}.yaml")
    else:
        _C.merge_from_file(f"{str(path)}/configs/simmim_pretrain_{args.dataset}.yaml")

    if args.lr:
        _C.TRAINING.BASE_LR = args.lr
    if args.mask_ratio:
        _C.DATA.MASK_RATIO = args.mask_ratio
    if args.train_frac:
        _C.DATA.TRAIN_FRAC = args.train_frac
    if args.load_checkpoint:
        _C.MODEL.ENCODER_CKPT_PATH = f"{_C.LOGGING.CKPT_DIR}/{args.load_checkpoint}"
    else:
        _C.TRAINING.FREEZE_WARMUP_EPOCHS = 0
    
    if args.cv:
        _C.TRAINING.CV_FOLDS = args.cv

    if args.name_suffix:
        _C.LOGGING.RUN_NAME = f"{_C.LOGGING.RUN_NAME}_{args.name_suffix}"

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