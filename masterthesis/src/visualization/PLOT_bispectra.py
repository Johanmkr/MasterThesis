"""
    DESCRIPTION OF MODULE:

    
"""

# Global imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Local imports
from .. import bispectrum
from .. import paths
from ..utils.figure import CustomFigure, SaveShow

# Temporary imports
from IPython import embed


def plot_triangle_space(
    gravity: str = "gr",
    seed: int = 1234,
    redshift: int = 1,
    k_res: int = 25,
    theta_res: int = 25,
    tag: str = "",
) -> None:
    path = paths.get_pickle_path(seed, gravity, redshift, k_res, theta_res, tag)
    df = pd.read_pickle(
        "/mn/stornext/d10/data/johanmkr/simulations/data_analysis/bispectra_dataframes/bispectrum_gr_s0_z1_k25_t25_tag_test_with_new_geomspace.pkl"
    )
    k1 = df["k1"].values.reshape(k_res, k_res, theta_res + 2)
    k2 = df["k2"].values.reshape(k_res, k_res, theta_res + 2)
    k3 = df["k3"].values.reshape(k_res, k_res, theta_res + 2)
    B = df["B"].values.reshape(k_res, k_res, theta_res + 2)

    # Temporary return function
    return k1, k2, k3, B


if __name__ == "__main__":
    plot_triangle_space()
