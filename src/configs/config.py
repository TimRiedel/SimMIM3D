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
_C.LOGGING.PROJECT_NAME = "SimMIM"
# Name of the run
_C.LOGGING.RUN_NAME = "run"
# Directory to store logs in
_C.LOGGING.LOGS_DIR = f"{HOME_DIR}/jobs/logs"
# Directory to save checkpoint to
_C.LOGGING.CKPT_DIR = f"{HOME_DIR}/jobs/checkpoints"


_C.DATA = CN()
# Directory in which the dataset is stored in
_C.DATA.DATA_DIR = f"{HOME_DIR}/datasets"
# Input image size in one dimension, must be divisible by patch size
_C.DATA.IMG_SIZE = 128
# Batch size
_C.DATA.BATCH_SIZE = 8
# Number of data loading threads
_C.DATA.NUM_WORKERS = 16
# [Pre-training] Ratio of masked patches to visible patches
_C.DATA.MASK_RATIO = 0.7
# [Fine-tuning] Number of classes for multiclass segmentation
_C.DATA.NUM_CLASSES = 4
# [Fine-tuning] Percentage of original training set to use
_C.DATA.TRAIN_FRAC = 1.0


_C.TRAINING = CN()
# Learning rate for the optimizer
_C.TRAINING.BASE_LR = 5e-4
# Weight decay for the optimizer
_C.TRAINING.WEIGHT_DECAY = 0.01
# [Fine-tuning] Number of warmup epochs in which encoder is frozen
_C.TRAINING.FREEZE_WARMUP_EPOCHS = 20
# Number of warmup epochs for learning rate scheduler
_C.TRAINING.LR_WARMUP_EPOCHS = 100
# Number of epochs to train for
_C.TRAINING.EPOCHS = 450
# [Fine-tuning] Number of cross validation folds to run
_C.TRAINING.CV_FOLDS = 1

_C.MODEL = CN()
# Number of input channels
_C.MODEL.IN_CHANNELS = 4
# Size of one patch
_C.MODEL.PATCH_SIZE = 16
# Dropout rate for the encoder
_C.MODEL.ENCODER_DROPOUT = 0.0
# [Fine-tuning] Path to checkpoint in which pre-trained weights for the encoder are stored
_C.MODEL.ENCODER_CKPT_PATH = ""


def get_config(args=None, pre_training=True):
    path = pathlib.Path(__file__).parent.resolve()

    if pre_training:
        _C.merge_from_file(f"{str(path)}/configs/simmim_pretrain_{args.dataset}.yaml")
    else:
        _C.merge_from_file(f"{str(path)}/configs/unetr_finetune_{args.dataset}.yaml")

    if args.project_name:
        _C.LOGGING.PROJECT_NAME = args.project_name
    if args.run_name:
        _C.LOGGING.run_name = args.run_name
    if args.ckpt_dir:
        _C.LOGGING.CKPT_DIR = args.ckpt_dir
    if args.logs_dir:
        _C.LOGGING.LOGS_DIR = args.logs_dir

    if args.lr:
        _C.TRAINING.BASE_LR = args.lr
    if args.mask_ratio:
        _C.DATA.MASK_RATIO = args.mask_ratio
    if args.train_frac:
        _C.DATA.TRAIN_FRAC = args.train_frac
    if args.cv:
        _C.TRAINING.CV_FOLDS = args.cv

    if args.load_model:
        _C.MODEL.ENCODER_CKPT_PATH = args.load_model
    else:
        _C.TRAINING.FREEZE_WARMUP_EPOCHS = 0

    return _C.clone()
