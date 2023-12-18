######### IMPORTS ###########################################
import os, sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from IPython import embed
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from src.utils import training_utils as tu
from src.models.RACOON import RACOON
from src.trainers.train_single_gpu import SingleGPUTrainer

# Params
data_params = {
    "train_test_split": [0.8, 0.2],
    "train_test_seeds": np.arange(0, 175, 1),
    "stride": 1,
    "redshift": 1.0,
    "random_seed": 42,
    "transforms": True,
}

model_params = {
    "input_size": (data_params["stride"], 256, 256),
    "layer_param": 64,
    "activation": nn.LeakyReLU(negative_slope=0.2, inplace=True),
    "output_activation": nn.Identity(),
    "bias": False,
    "dropout": 0.5,
}

loader_params = {
    "batch_size": int((256 * 3) * 3.6),
    "num_workers": 32,
    "prefetch_factor": 32,
}

optimizer_params = {
    "lr": 0.1,
    "betas": (0.5, 0.999),
    "weight_decay": 1e-11,
}

training_params = {
    "epochs": 10,
    "breakout_loss": 1e-3,
    "tol": 0.5,
}


# 1. make datasets
train_data, test_data = tu.make_training_and_testing_data(**data_params)

# 2. make model
model = RACOON(**model_params)

# 3. optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
loss_fn = nn.BCEWithLogitsLoss()

# 4. trainer
trainer = SingleGPUTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_dataset=train_data,
    test_dataset=test_data,
    test_name="test",
    **loader_params,
)

# 5. train
trainer.train(**training_params)
# while input("Train 1 epoch? (y/n): ") == "y":
#     trainer.train(epochs=1, tol=0.5)
# if input("Embed? (y/n): ") == "y":
#     embed()
