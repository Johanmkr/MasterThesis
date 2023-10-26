import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# add path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from src.visualization import PLOT_cube_slice as pcs
from src.visualization import PLOT_power_spectra as pps


# Make cube slices plot
# pcs.cube_slices(seed=1234, cmap="plasma")


# Plot matter power spectra from CLASS
# pps.class_matter_power_spectra()
pps.average_matter_power_spectra()
pps.average_potential_power_spectra()
