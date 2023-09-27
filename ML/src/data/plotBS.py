import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure import CustomFigure
import bispectrum as bs
import powerspectra as ps
import pandas as pd

from IPython import embed


## Start with some easy testing to see if things work:

seed = 1234

datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"

grBispectrum = bs.CubeBispectrum(datapath + f"seed{seed:04d}/gr/gr_000_phi.h5")
newtonBispectrum = bs.CubeBispectrum(datapath + f"seed{seed:04d}/newton/newton_000_phi.h5")

def plot_equilateral(k_range):
    grBispectrum.equilateral_bispectrum(k_range).plot(x="k", y="B", label="GR")
    newtonBispectrum.equilateral_bispectrum(k_range).plot(x="k", y="B", label="Newton")
    plt.title("Equilateral Bispectrum")
    plt.ylabel(r"$B(k_1, k_2, k_3)$")
    plt.xlabel(r"$k_1 = k_2 = k_3$ [h/Mpc]")
    plt.legend()
    plt.show()