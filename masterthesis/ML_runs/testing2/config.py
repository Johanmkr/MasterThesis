# Configurations file for the ML pipeline
import numpy as np
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim

# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from src.data.transforms import Normalise
from src.models import SLOTH, MOTH

VERBOSE = True

DATA_PARAMS = {
    "train_test_val_split": (0.8, 0.1, 0.1),
    "batch_size": 8,
    "num_workers": 64,
    "stride": 256,
    "redshifts": 1.0,
    "transform": Normalise(),
    "additional_info": False,
    "total_seeds": np.arange(0, 500, 1),
    "random_seed": 42,
    "prefetch_factor": 64,
    "nr_train_loaders": 1,
}


MODEL_PARAMS = {
    "input_size": (DATA_PARAMS["stride"], 256, 256),
    "layer_param": 16,
    "activation": nn.LeakyReLU(),
    "output_activation": nn.Sigmoid(),
    "bias": False,
    "dropout": 0.25,
}


TRAINING_PARAMS = {
    "epochs": 10,
}

##### MODEL #####
MODEL = SLOTH.SLOTH(**MODEL_PARAMS)


##### LOSS #####
LOSS_FN = nn.BCELoss()


##### OPTIMIZER #####
OPTIMIZER_PARAMS = {
    "lr": 0.0002,
    "betas": (0.5, 0.999),
    "weight_decay": 1e-11,
}
OPTIMIZER = optim.Adam(MODEL.parameters(), **OPTIMIZER_PARAMS)
