MULTIPLE_GPUS = False
print(f"MULTIPLE_GPUS: {MULTIPLE_GPUS}")
VERBOSE = True

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

if MULTIPLE_GPUS:
    from src.trainers.train_multiple_gpu import MultipleGPUTrainer as TRAINER
else:
    from src.trainers.train_single_gpu import SingleGPUTrainer as TRAINER

import config as cfg

# print all info from config file
if VERBOSE:
    print("CONFIGURATION:")
    for dicti in [
        cfg.data_params,
        cfg.model_params,
        cfg.loader_params,
        cfg.optimizer_params,
        cfg.training_params,
    ]:
        for key, value in dicti.items():
            print(f"{key}: {value}")

# 1. make datasets
train_data, test_data = tu.make_training_and_testing_data(**cfg.data_params)

# 2. make model
model = RACOON(**cfg.model_params)
if VERBOSE:
    print(model)

# 3. trainer
trainer = TRAINER(
    model=model,
    train_dataset=train_data,
    test_dataset=test_data,
    test_name=f"RACCOON_test_lp{cfg.model_params['layer_param']}_lr{cfg.optimizer_params['lr']}",
    **cfg.loader_params,
)

# 4. train
trainer.run(**cfg.training_params, optimizer_params=cfg.optimizer_params)
