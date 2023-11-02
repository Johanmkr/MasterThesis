import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# add path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from src.data import transforms, custom_dataset

from IPython import embed


# norm = transforms.Normalise(redshifts=[1.0])
ds = custom_dataset.CustomDataset(transform=None, additional_info=True)

# embed()
