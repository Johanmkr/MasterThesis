# CONFIGURATION FILE FOR TESTING
import numpy as np


MODEL_NAME = "MOUSE"

# Data parameters
NUM_WORKERS = 4
NGRID = 256
STRIDE = 2
SEEDS = np.arange(0, 2000, 50)
REDSHIFTS = 1.0
AXES = [0]

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0.0005
TRAIN_TEST_VAL_SPLIT = (0.6, 0.2, 0.2)
SPLIT_SEED = 42

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]  # -1 for all available GPUs, 0 for CPU
PRECISION = 32

# Logging
LOG_DIR = "tb_logs"
