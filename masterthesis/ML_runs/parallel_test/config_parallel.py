""" 
    Configuration file for parallel network test
"""

import numpy as np
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim

# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from src.models.PENGUIN import PENGUIN
from src.models.train_model import ModelTrainer

VERBOSE = True
