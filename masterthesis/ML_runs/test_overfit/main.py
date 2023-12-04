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
import pandas as pd
import time

######### ADD PARENT DIRECTORY TO PATH #########################
# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)


######### LOCAL IMPORTS ########################################
from src.data.whole_cube_dataset import WholeCubeDataset, make_whole_dataset
from src.models.SLOTH import SLOTH  # Model of 3D conv network
from src.models.train_model import train_model, overfit_model

######### IMPORT CONFIG FILE ####################################
import overfit_config as cfg


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
train_loader, test_loader, val_loader = make_whole_dataset(**cfg.DATA_PARAMS)
if cfg.VERBOSE:
    print("Data loaded.\n\n")


######### MODEL ###############################################
# if cfg.VERBOSE:
#     print("Loading model...")
# model = cfg.MODEL
# if cfg.VERBOSE:
#     print("Model loaded.\n\n")


######### SUMMARY ###############################################
if cfg.VERBOSE and GPU:
    print("Printing summary")
    print(f"Model: {model}")


######### OPTIMIZER ###############################################
# optimizer = cfg.OPTIMIZER
loss_fn = cfg.LOSS_FN

######### TRAINING ###############################################
# train_model(
#     model,
#     optimizer,
#     loss_fn,
#     train_loaders[0],
#     val_loader,
#     10,
#     device,
#     verbose=True,
# )

######### OVERFIT ###############################################


def run_overfit(
    model,
    optimizer,
    loss_fn,
    train_loader,
    device,
    activation,
    layer_param,
    dropout,
    verbose=True,
    tol: float = 1e-2,
    info_frame=None,
):
    start_time = time.time()
    epochs, train_losses, train_accs = overfit_model(
        model,
        optimizer,
        loss_fn,
        train_loader,
        device,
        verbose,
        tol,
    )
    end_time = time.time()
    dF = pd.DataFrame(
        {
            "epochs": epochs,
            "train_losses": train_losses,
            "train_accs": train_accs,
            "activation": activation,
            "layer_param": layer_param,
            "dropout": dropout,
            "duration_sec": end_time - start_time,
            "duration_min": (end_time - start_time) / 60,
        }
    )
    dF.to_csv(f"outputs/overfit_{activation}_{layer_param}_{dropout}.csv")


batch1 = ["ReLU", "LeakyReLU"]
batch2 = ["ELU", "GELU"]
batch3 = ["SELU", "SiLU"]


def find_for_batch(batch):
    for activation in batch:
        for layer_param in cfg.layer_params.keys():
            for dropout in cfg.dropouts.keys():
                cfg.MODEL_PARAMS["activation"] = cfg.activations[activation]
                cfg.MODEL_PARAMS["layer_param"] = cfg.layer_params[layer_param]
                cfg.MODEL_PARAMS["dropout"] = cfg.dropouts[dropout]
                model = SLOTH(**cfg.MODEL_PARAMS)
                optimizer = optim.Adam(model.parameters(), **cfg.OPTIMIZER_PARAMS)
                print(
                    f"Running for:\n activation: {activation}\n layer_param: {layer_param}\n dropout: {dropout}\n--------------------------------\n"
                )
                print(f"Model: {model}")
                run_overfit(
                    model,
                    optimizer,
                    loss_fn,
                    train_loader,
                    device,
                    activation,
                    layer_param,
                    dropout,
                    verbose=True,
                    tol=1e-2,
                )


if __name__ == "__main__":
    BATCH = sys.argv[1]
    if BATCH == "1":
        find_for_batch(batch1)
    elif BATCH == "2":
        find_for_batch(batch2)
    elif BATCH == "3":
        find_for_batch(batch3)
    else:
        print("Invalid batch number")
        sys.exit(1)
