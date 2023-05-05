# Training hyperparameters
INPUT_SIZE = (128, 128, 128)
NUM_CLASSES = 4
LEARNING_RATE = 1e-2
BATCH_SIZE = 4
NUM_EPOCHS = 100

# Dataset
DATA_DIR = None #TODO: Choose data directory
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0] #TODO: Set number of GPUs to use
PRECISION = 16 #TODO: Choose precision