# Training hyperparameters
INPUT_SIZE = (96, 96, 96)
NUM_CHANNELS = 4
NUM_CLASSES = 3
PATCH_SIZE = 16
LEARNING_RATE = 3e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100 #TODO: Set number of epochs

# Dataset
BRATS_DATA_DIR = "/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
STRATEGY="ddp"
DEVICES = "-1"
NODES = 1
PRECISION = "16-mixed"

# Logging
WANDB_DIR = "/dhc/home/tim.riedel/bachelor-thesis/jobs/wandb"