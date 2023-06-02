# Logging
RUN_NAME = "SimMIM3D_PT_Brats"
WANDB_DIR = "/dhc/home/tim.riedel/bachelor-thesis/jobs/wandb"
CHECKPOINT_DIR = f"/dhc/home/tim.riedel/bachelor-thesis/jobs/checkpoints/{RUN_NAME}"

# Training hyperparameters
IMG_SIZE = (96, 96, 96)
IN_CHANNELS = 4
NUM_CLASSES = 3
PATCH_SIZE = 16 # TODO: higher patch size reduces computation needs
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
NUM_EPOCHS = 2
ENCODER_DROPOUT = 0.1

# Dataset
BRATS_DATA_DIR = "/dhc/home/tim.riedel/bachelor-thesis/data/BraTS2017"
NUM_WORKERS = 16

# Compute related
ACCELERATOR = "gpu"
STRATEGY="ddp"
DEVICES = "-1"
NODES = 1
PRECISION = "16-mixed"
