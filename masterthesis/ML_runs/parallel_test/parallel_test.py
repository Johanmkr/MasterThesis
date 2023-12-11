""" 
This file is used to test the parallelization of the code, across the GPUs
"""

######### STUFF ###########################################

# stuff

###############################################################

######### IMPORTS ###########################################
import os, sys
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from src.data.cube_datasets import SlicedCubeDataset
from src.models.PENGUIN import PENGUIN

###############################################################
######### VARIABLES ###########################################
# Data variables
train_test_split = (0.8, 0.2)
batch_size = 256 * 3 * 2  # corresponds to 2 cubes
num_workers = 64
redshift = 1.0
stride = 1
train_test_seeds = np.arange(0, 10, 1)
val_seeds = np.arange(1750, 1755, 1)
prefetch_factor = 32
random_seed = 42

# Model variables
input_size = (stride, 256, 256)
layer_param = 64
activation = nn.ReLU()
output_activation = nn.Sigmoid()
bias = False
dropout = 0.5

# Training variables
lr = 0.001
betas = (0.9, 0.999)
weight_decay = 1e-5

###############################################################
######### LOAD DATA ###########################################

random.seed(random_seed)
random.shuffle(train_test_seeds)

array_length = len(train_test_seeds)
assert (
    abs(sum(train_test_split) - 1.0) < 1e-6
), "Train and test split does not sum to 1."
train_length = int(array_length * train_test_split[0])
test_length = int(array_length * train_test_split[1])
train_seeds = train_test_seeds[:train_length]
test_seeds = train_test_seeds[train_length:]

# Make datasets
print("Making datasets...")
print(f"Training set: {len(train_seeds)} seeds")
train_dataset = SlicedCubeDataset(
    stride=stride,
    redshift=redshift,
    seeds=train_seeds,
)
print(f"Test set: {len(test_seeds)} seeds")
test_dataset = SlicedCubeDataset(
    stride=stride,
    redshift=redshift,
    seeds=test_seeds,
)


###############################################################
######### MODEL ###############################################

model = PENGUIN(
    input_size=input_size,
    layer_param=layer_param,
    activation=activation,
    output_activation=output_activation,
    bias=bias,
    dropout=dropout,
)


###############################################################
######### GPU stuff ###########################################
GPU = torch.cuda.is_available()
world_size = torch.cuda.device_count()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


###############################################################
######### TRAINING (PARALLEL) #################################


def train(rank, world_size, train_dataset, model):
    setup(rank, world_size)

    # Make dataloaders
    print("Making dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
        pin_memory=True,
    )

    # Make model distributed
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(
        ddp_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )

    # Train model
    for epoch in range(1):
        for batch in train_dataloader:
            print(f"Rank {rank} got batch {batch}")
            images, labels = batch["image"], batch["label"]
            images = images.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            if 

    cleanup()


###############################################################
