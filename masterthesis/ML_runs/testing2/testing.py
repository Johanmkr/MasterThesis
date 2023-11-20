import numpy as np
import sys, os
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from IPython import embed

# Local imports
from src.data.custom_dataset import CustomDataset, make_dataset
from src.models.MOTH import MOTH  # Dummy model of convolutional network
import config as cfg


train_loader, test_loader, val_loader = make_dataset(**cfg.DATA_PARAMS)

total_batches = len(train_loader)
total_time = 0.0
set_time = time.time()
for i, batch in enumerate(train_loader):
    # start_time = time.time()

    images, labels = batch["image"], batch["label"]

    end_time = time.time()
    batch_time = end_time - set_time
    total_time += batch_time
    set_time = end_time

    print(f"Batch {i+1}/{total_batches} took {batch_time:.4f} seconds.")

average_time = total_time / total_batches
print(f"Average time per batch: {average_time:.4f} seconds.")
print(f"Total time: {total_time:.4f} seconds.")
