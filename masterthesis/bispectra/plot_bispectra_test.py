import numpy as np
import matplotlib.pyplot as plt
import calculate_bispectrum as cb
import os, sys
import h5py

# add path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

