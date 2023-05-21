import os

import torch

# Training hyperparameters
INPUT_SIZE = (64, 64, 64)
NUM_CHANNELS = 4
NUM_CLASSES = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 10 #TODO: Set number of epochs

# Dataset
DATA_DIR = "/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = 2 #TODO: Set number of GPUs to use
PRECISION = "16-mixed"