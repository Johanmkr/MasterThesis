import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure import CustomFigure
import bispectrum as bs
import powerspectra as ps
import cube
import pandas as pd

from IPython import embed


## Start with some easy testing to see if things work:

seed = 1234
redshift=1

datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"

grBispectrum = bs.CubeBispectrum(datapath + f"seed{seed:04d}/gr/gr_{cube.redshift_to_snap[redshift]}_phi.h5")
newtonBispectrum = bs.CubeBispectrum(datapath + f"seed{seed:04d}/newton/newton_{cube.redshift_to_snap[redshift]}_phi.h5")

def plot_equilateral(k_range):
    grdBk = grBispectrum.equilateral_bispectrum(k_range)
    ndBk = newtonBispectrum.equilateral_bispectrum(k_range)
    eqfig = CustomFigure()
    settings = {
                "xscale": "log",
                "yscale": "log",
                "xlabel": r"k $[h/Mpc]$",
                "ylabel": r"P(k) $[h/Mpc]^3$",
                "title": f"Equilateral Bispectrum of 'phi' at redshift z={redshift:.1f}, for seed {seed:04d}"
            }
    eqfig.set_settings(settings)

    lines = []

    lines.append(Line2D(grdBk["k"], abs(grdBk["B"]), color="blue", label="GR"))
    lines.append(Line2D(ndBk["k"], abs(ndBk["B"]), color="red", label="Newton"))

    for line in lines:
        eqfig.ax.add_line(line)

    leg1 = eqfig.ax.legend(handles=lines, loc="upper right")
    eqfig.ax.add_artist(leg1)
    eqfig.ax.autoscale_view()

    comparison_axis_settings = {
        # "xscale": "log",
        "yscale": "log",
        "ylabel": r"GR-Newton",
    }
    eqfig.gen_twinx(comparison_axis_settings)

    eqfig.axes[1].plot(grdBk["k"], grdBk["B"] - ndBk["B"], color="black", label="GR-Newton")

def plot_squeezed(k_range):
    grdBk = grBispectrum.squeezed_bispectrum(k_range)
    ndBk = newtonBispectrum.squeezed_bispectrum(k_range)
    eqfig = CustomFigure()
    settings = {
                "xscale": "log",
                "yscale": "log",
                "xlabel": r"k $[h/Mpc]$",
                "ylabel": r"P(k) $[h/Mpc]^3$",
                "title": f"Squeezed Bispectrum of 'phi' at redshift z={redshift:.1f}, for seed {seed:04d}"
            }
    eqfig.set_settings(settings)

    lines = []

    lines.append(Line2D(grdBk["k"], abs(grdBk["B"]), color="blue", label="GR"))
    lines.append(Line2D(ndBk["k"], abs(ndBk["B"]), color="red", label="Newton"))

    for line in lines:
        eqfig.ax.add_line(line)

    leg1 = eqfig.ax.legend(handles=lines, loc="upper right")
    eqfig.ax.add_artist(leg1)
    eqfig.ax.autoscale_view()

    comparison_axis_settings = {
        # "xscale": "log",
        "yscale": "log",
        "ylabel": r"GR-Newton",
    }
    eqfig.gen_twinx(comparison_axis_settings)

    eqfig.axes[1].plot(grdBk["k"], grdBk["B"] - ndBk["B"], color="black", label="GR-Newton")


ks = np.geomspace(grBispectrum.kF, grBispectrum.kN, 75)
plot_equilateral(ks)
plot_squeezed(ks)
plt.show()
