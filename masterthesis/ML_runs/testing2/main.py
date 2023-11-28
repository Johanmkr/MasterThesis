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
from src.models.train_model import train_model, overfit_model

######### IMPORT CONFIG FILE ####################################
import config as cfg


######### GPU STUFF ###########################################
if cfg.VERBOSE:
    print("Checking for GPU...")
GPU = torch.cuda.is_available()
device = torch.device("cuda:1" if GPU else "cpu")
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
model = cfg.MODEL
if cfg.VERBOSE:
    print("Model loaded.\n\n")


######### SUMMARY ###############################################
if cfg.VERBOSE and GPU:
    print("Printing summary")
    print(f"Model: {model}")


######### OPTIMIZER ###############################################
optimizer = cfg.OPTIMIZER
loss_fn = cfg.LOSS_FN

######### TRAINING ###############################################
train_model(
    model,
    optimizer,
    loss_fn,
    train_loaders[0],
    val_loader,
    10,
    device,
    verbose=True,
)

######### OVERFIT ###############################################
# overfit_model(
#     model,
#     optimizer,
#     loss_fn,
#     train_loaders[0],
#     device,
#     verbose=True,
#     tol=1e-1,
# )
