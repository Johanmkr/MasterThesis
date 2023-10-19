import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import pandas as pd
import yaml
from tqdm import tqdm

# Local imports
from ..data import cube
from ..utils import paths
from . import bispectrum

# from . import pyliansPK

# Temporary imports
from IPython import embed

class AVG_bispectrum:
    def __init__(self, z:float=1.0):
        self.seeds = np.arange(0, 1501, 50)