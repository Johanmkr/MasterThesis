import numpy as np
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim

# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from src.data.transforms import Normalise
from src.models import SLOTH

# Varying parameters:
activations = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(negative_slope=0.1),
    "ELU": nn.ELU(),
    "GELU": nn.GELU(),
    "SELU": nn.SELU(),
    "SiLU": nn.SiLU(),
}

layer_params = {
    "small": 5,
    "medium": 15,
    "large": 25,
}

dropouts = {
    "quarter": 0.25,
    "half": 0.5,
}

VERBOSE = True

DATA_PARAMS = {
    "train_test_val_split": (0.7, 0.2, 0.1),
    "batch_size": 16,
    "num_workers": 36,
    "redshifts": 1.0,
    "transform": Normalise(redshifts=1.0),
    "total_seeds": np.arange(0, 2000, 1),
    "random_seed": 42,
    "prefetch_factor": 64,
}

MODEL_PARAMS = {
    "input_size": (256, 256, 256),
    "output_activation": nn.Sigmoid(),
    "bias": False,
}

LOSS_FN = nn.BCELoss()

OPTIMIZER_PARAMS = {
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-8,
}
