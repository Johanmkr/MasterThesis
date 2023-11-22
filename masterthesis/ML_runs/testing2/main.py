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


# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from IPython import embed

# Local imports
from src.data.custom_dataset import CustomDataset, make_dataset
from src.models.MOTH import MOTH  # Dummy model of convolutional network
from src.models.SLOTH import SLOTH  # Model of 3D conv network

verbose = True

######### GPU STUFF #############################################
if verbose:
    print("Checking for GPU...")
GPU = torch.cuda.is_available()
# GPU = False
device = torch.device("cuda:0" if GPU else "cpu")
print("GPU: ", GPU)
print("Device: ", device)
cudnn.benchmark = True
if verbose:
    print("GPU check complete.\n\n")
#################################################################


######### DATA ###############################################
if verbose:
    print("Loading data...")
train_loader, test_loader, val_loader = make_dataset(
    train_test_val_split=(0.8, 0.1, 0.1),
    batch_size=8,
    num_workers=64,
    stride=256,
    redshifts=1.0,
    additional_info=False,
    total_seeds=np.arange(0, 500, 1),
    random_seed=42,
    prefetch_factor=64,
)
if verbose:
    print("Data loaded.\n\n")


######### MODEL ###############################################
if verbose:
    print("Loading model...")
model = SLOTH(
    input_size=(256, 256, 256),
    layer_param=16,
).to(device)
# model = parallel.DistributedDataParallel(
# model, device_ids=[device], process_group=dist.new_group()
# )
if verbose:
    print("Model loaded.\n\n")


######### SUMMARY ###############################################
if verbose and GPU:
    print("Printing summary")
    model.printSummary()


######### OPTIMIZER ###############################################
optimizer = optim.Adam(
    model.parameters(), lr=1e-1, betas=(0.5, 0.999), weight_decay=1e-5
)
loss_fn = nn.BCELoss().to(device)
# data = next(iter(train_loader))

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
