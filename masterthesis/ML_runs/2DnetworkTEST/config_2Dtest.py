""" 
    Configuration file for 2D network test
"""

import numpy as np
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from src.data.transforms import Normalise
from src.models.PENGUIN import PENGUIN
from src.models.train_model import ModelTrainer

VERBOSE = True

DATA_PARAMS = {
    "train_test_val_split": (0.7, 0.2, 0.1),
    "batch_size": 256,
    "num_workers": 96,
    "stride": 1,
    "redshifts": 1.0,
    "transform": Normalise(),
    "total_seeds": np.arange(0, 1000, 1),
    "random_seed": 42,
    "prefetch_factor": 256,
}


MODEL_PARAMS = {
    "input_size": (DATA_PARAMS["stride"], 256, 256),
    "layer_param": 15,
    "activation": nn.LeakyReLU(),
    "output_activation": nn.Sigmoid(),
    "bias": False,
    "dropout": 0.25,
}


##### MODEL #####
MODEL = PENGUIN(**MODEL_PARAMS)


##### LOSS #####
LOSS_FN = nn.BCELoss()

##### OPTIMIZER #####
OPTIMIZER_PARAMS = {
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-5,
}
OPTIMIZER = optim.Adam(MODEL.parameters(), **OPTIMIZER_PARAMS)
