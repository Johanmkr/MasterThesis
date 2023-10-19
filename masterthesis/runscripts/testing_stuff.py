import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# add path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)
# parent_dir

#   Local imports
from src.features import analytical_bispectrum
from src.features import classPK

from IPython import embed


if __name__ == "__main__":
    test = analytical_bispectrum.AnalyticalBispectrum(
        k_range=np.geomspace(1e-5, 1e-1, 1000)
    )
    plt.loglog(test.k_range, test.B_equilateral, label="equilateral")
    plt.loglog(test.k_range, test.B_squeezed, label="squeezed")
    plt.show()
