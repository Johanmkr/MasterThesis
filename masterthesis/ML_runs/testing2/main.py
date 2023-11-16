import numpy as np
import sys, os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt


# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

# Local imports
from src.data.custom_dataset import CustomDataset, make_dataset
from src.models.MOTH import MOTH  # Dummy model of convolutional network


######### GPU STUFF #############################################
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("GPU: ", GPU)
print("Device: ", device)
cudnn.benchmark = True
#################################################################


train_loader, test_loader, val_loader = make_dataset(
    train_test_val_split=(0.8, 0.1, 0.1),
    batch_size=32,
    num_workers=4,
    stride=2,
    redshifts=1.0,
    additional_info=True,
    total_seeds=np.arange(0, 2000, 50),
)


model = MOTH(
    input_size=(2, 256, 256),
    layer_param=1,
)

model.printSummary()
