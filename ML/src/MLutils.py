"""
    Utility file for the Machine Learning pipeline (and other stuhf)
"""

#   Imports
import numpy as np
import matplotlib.pyplot as plt 
import h5py
import os
import matplotlib.cm as cm
import configparser

# torch stuff
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as ts
from torchsummary import summary
generator1 = torch.Generator().manual_seed(42)

# for visualisation loop:
from matplotlib import colors