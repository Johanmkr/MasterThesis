######### GLOBAL IMPORTS #######################################
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
from IPython import embed

######### ADD PARENT DIRECTORY TO PATH #########################
# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)


######### LOCAL IMPORTS ########################################
from src.data.custom_dataset import CustomDataset, make_dataset
from src.models.MOTH import MOTH  # Dummy model of convolutional network
from src.models.SLOTH import SLOTH  # Model of 3D conv network

######### IMPORT CONFIG FILE ####################################
import config as cfg


######### GPU STUFF ###########################################
if cfg.VERBOSE:
    print("Checking for GPU...")
GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if GPU else "cpu")
print("GPU: ", GPU)
print("Device: ", device)
cudnn.benchmark = True
if cfg.VERBOSE:
    print("GPU check complete.\n\n")


######### DATA ###############################################
if cfg.VERBOSE:
    print("Loading data...")
train_loaders, test_loader, val_loader = make_dataset(**cfg.DATA_PARAMS)
if cfg.VERBOSE:
    print("Data loaded.\n\n")


######### MODEL ###############################################
if cfg.VERBOSE:
    print("Loading model...")
model = cfg.MODEL.to(device)
if cfg.VERBOSE:
    print("Model loaded.\n\n")


######### SUMMARY ###############################################
if cfg.VERBOSE and GPU:
    print("Printing summary")
    model.printSummary()


######### OPTIMIZER ###############################################
optimizer = cfg.OPTIMIZER
loss_fn = cfg.LOSS_FN.to(device)

######### TRAINING ###############################################
tot_len = len(train_loader)
running_loss = 0
for epoch in range(1, 21):
    model.train()
    # for i in range(11):
    for i, data in enumerate(train_loader):
        # Get the inputs
        print(f"Epoch: {epoch}, Batch: [{i+1}/{tot_len}]")
        images, labels = data["image"], data["label"]
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # embed()

        # Forward + backward + optimize
        # optimizer.zero_grad()
        outputs = model(images)
        # embed()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10th mini-batches
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {running_loss/10}")
            running_loss = 0.0
