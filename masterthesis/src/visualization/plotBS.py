"""
    DESCRIPTION OF MODULE:

    
"""

# Global imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# # Local imports
from src.utils.figure import CustomFigure
# from .. import bispectrum as bs
# from .. import powerspectra as ps
# from .. import cube

# # Temporary imports
# from IPython import embed


# ## Start with some easy testing to see if things work:

# seed = 1234
# redshift=1

# datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"

# grBispectrum = bs.CubeBispectrum(datapath + f"seed{seed:04d}/gr/gr_{cube.redshift_to_snap[redshift]}_phi.h5")
# newtonBispectrum = bs.CubeBispectrum(datapath + f"seed{seed:04d}/newton/newton_{cube.redshift_to_snap[redshift]}_phi.h5")

# def plot_equilateral(k_range):
#     grdBk = grBispectrum.equilateral_bispectrum(k_range)
#     ndBk = newtonBispectrum.equilateral_bispectrum(k_range)
#     eqfig = CustomFigure()
#     settings = {
#                 "xscale": "log",
#                 "yscale": "log",
#                 "xlabel": r"k $[h/Mpc]$",
#                 "ylabel": r"P(k) $[h/Mpc]^3$",
#                 "title": f"Equilateral Bispectrum of 'phi' at redshift z={redshift:.1f}, for seed {seed:04d}"
#             }
#     eqfig.set_settings(settings)

#     lines = []

#     lines.append(Line2D(grdBk["k"], abs(grdBk["B"]), color="blue", label="GR"))
#     lines.append(Line2D(ndBk["k"], abs(ndBk["B"]), color="red", label="Newton"))

#     for line in lines:
#         eqfig.ax.add_line(line)

#     leg1 = eqfig.ax.legend(handles=lines, loc="upper right")
#     eqfig.ax.add_artist(leg1)
#     eqfig.ax.autoscale_view()

#     comparison_axis_settings = {
#         # "xscale": "log",
#         "yscale": "log",
#         "ylabel": r"GR-Newton",
#     }
#     eqfig.gen_twinx(comparison_axis_settings)

#     eqfig.axes[1].plot(grdBk["k"], grdBk["B"] - ndBk["B"], color="black", label="GR-Newton")

# def plot_squeezed(k_range):
#     grdBk = grBispectrum.squeezed_bispectrum(k_range)
#     ndBk = newtonBispectrum.squeezed_bispectrum(k_range)
#     eqfig = CustomFigure()
#     settings = {
#                 "xscale": "log",
#                 "yscale": "log",
#                 "xlabel": r"k $[h/Mpc]$",
#                 "ylabel": r"P(k) $[h/Mpc]^3$",
#                 "title": f"Squeezed Bispectrum of 'phi' at redshift z={redshift:.1f}, for seed {seed:04d}"
#             }
#     eqfig.set_settings(settings)

#     lines = []

#     lines.append(Line2D(grdBk["k"], abs(grdBk["B"]), color="blue", label="GR"))
#     lines.append(Line2D(ndBk["k"], abs(ndBk["B"]), color="red", label="Newton"))

#     for line in lines:
#         eqfig.ax.add_line(line)

#     leg1 = eqfig.ax.legend(handles=lines, loc="upper right")
#     eqfig.ax.add_artist(leg1)
#     eqfig.ax.autoscale_view()

#     comparison_axis_settings = {
#         # "xscale": "log",
#         "yscale": "log",
#         "ylabel": r"GR-Newton",
#     }
#     eqfig.gen_twinx(comparison_axis_settings)

#     eqfig.axes[1].plot(grdBk["k"], grdBk["B"] - ndBk["B"], color="black", label="GR-Newton")

def plot_equilateral_bispectra_from_pre_calculated_seed(seed, redshift):
    grdBk = pd.read_csv(f"../pre_computed_bispectra/seed{seed:04d}_gr_equilateral_rs{redshift:04d}.csv")
    ndBk = pd.read_csv(f"../pre_computed_bispectra/seed{seed:04d}_newton_equilateral_rs{redshift:04d}.csv")
    eqfig = CustomFigure()
    settings = {
                "xscale": "log",
                "yscale": "log",
                "xlabel": r"k $[h/Mpc]$",
                "ylabel": r"B(k,k,k) $[h/Mpc]^3$",
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
        "ylabel": r"|(GR-Newton)/Newton|",
    }
    ax2 = eqfig.gen_twinx(comparison_axis_settings)
    
    comparison = Line2D(grdBk["k"], abs((grdBk["B"] - ndBk["B"])/ndBk["B"]), color="black", label="Difference")
    eqfig.axes[1].add_line(comparison)
    eqfig.axes[1].autoscale_view()
    
    leg2 = eqfig.axes[1].legend(handles=[comparison], loc="lower right")

    eqfig.fig.savefig(f"../temp_figures/seed{seed:04d}_equilateral_rs{redshift:04d}.png")
    plt.close(eqfig.fig)
    return grdBk, ndBk

def plot_squeeze_bispectra_from_pre_calculated_seed(seed, redshift):
    grdBk = pd.read_csv(f"../pre_computed_bispectra/seed{seed:04d}_gr_squeezed_rs{redshift:04d}.csv")
    ndBk = pd.read_csv(f"../pre_computed_bispectra/seed{seed:04d}_newton_squeezed_rs{redshift:04d}.csv")
    eqfig = CustomFigure()
    settings = {
                "xscale": "log",
                "yscale": "log",
                "xlabel": r"k $[h/Mpc]$",
                "ylabel": r"B(k,k,$\theta=19\pi/20$) $[h/Mpc]^3$",
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
        "ylabel": r"|(GR-Newton)/Newton|",
    }
    ax2 = eqfig.gen_twinx(comparison_axis_settings)
    
    comparison = Line2D(grdBk["k"], abs((grdBk["B"] - ndBk["B"])/ndBk["B"]), color="black", label="Difference")
    eqfig.axes[1].add_line(comparison)
    eqfig.axes[1].autoscale_view()
    
    leg2 = eqfig.axes[1].legend(handles=[comparison], loc="lower right")

    eqfig.fig.savefig(f"../temp_figures/seed{seed:04d}_squeezed_rs{redshift:04d}.png")
    plt.close(eqfig.fig)
    return grdBk, ndBk

# ks = np.geomspace(grBispectrum.kF, grBispectrum.kN, 75)
# plot_equilateral(ks)
# plot_squeezed(ks)

seeds = [0,50,100,200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
redshifts = [0, 1, 5, 15]

average_equilateral_gr = pd.DataFrame({"k": np.geomspace(2*np.pi/5120, 1e-2, 1000), "B": np.zeros(1000), "Q": np.zeros(1000)})
average_equilateral_newton = pd.DataFrame({"k": np.geomspace(2*np.pi/5120, 1e-2, 1000), "B": np.zeros(1000), "Q": np.zeros(1000)})
average_squeezed_gr = pd.DataFrame({"k": np.geomspace(2*np.pi/5120, 1e-2, 1000), "B": np.zeros(1000), "Q": np.zeros(1000)})
average_squeezed_newton = pd.DataFrame({"k": np.geomspace(2*np.pi/5120, 1e-2, 1000), "B": np.zeros(1000), "Q": np.zeros(1000)})


for seed in seeds:
    for redshift in redshifts:
        eqGR, eqN = plot_equilateral_bispectra_from_pre_calculated_seed(seed, redshift)
        sqGR, sqN = plot_squeeze_bispectra_from_pre_calculated_seed(seed, redshift)
        if redshift == 1:
            average_equilateral_gr["B"] += eqGR["B"]
            average_equilateral_gr["Q"] += eqGR["Q"]
            average_equilateral_newton["B"] += eqN["B"]
            average_equilateral_newton["Q"] += eqN["Q"]
            average_squeezed_gr["B"] += sqGR["B"]
            average_squeezed_gr["Q"] += sqGR["Q"]
            average_squeezed_newton["B"] += sqN["B"]
            average_squeezed_newton["Q"] += sqN["Q"]

average_equilateral_gr["B"] /= len(seeds)
average_equilateral_gr["Q"] /= len(seeds)
average_equilateral_newton["B"] /= len(seeds)
average_equilateral_newton["Q"] /= len(seeds)
average_squeezed_gr["B"] /= len(seeds)
average_squeezed_gr["Q"] /= len(seeds)
average_squeezed_newton["B"] /= len(seeds)
average_squeezed_newton["Q"] /= len(seeds)

average_equilateral_fig = CustomFigure()
settings = {
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"k $[h/Mpc]$",
            "ylabel": r"B(k,k,k) $[h/Mpc]^3$",
            "title": f"Average Equilateral Bispectra of 'phi' at redshift z=1"
        }
average_equilateral_fig.set_settings(settings)

lines = []
lines.append(Line2D(average_equilateral_gr["k"], abs(average_equilateral_gr["B"]), color="blue", label="GR"))
lines.append(Line2D(average_equilateral_newton["k"], abs(average_equilateral_newton["B"]), color="red", label="Newton"))

for line in lines:
    average_equilateral_fig.ax.add_line(line)

leg1 = average_equilateral_fig.ax.legend(handles=lines, loc="upper right")

average_equilateral_fig.ax.add_artist(leg1)
average_equilateral_fig.ax.autoscale_view()

comparison_axis_settings = {
        # "xscale": "log",
        "yscale": "log",
        "ylabel": r"|GR-Newton|",
    }
ax2 = average_equilateral_fig.gen_twinx(comparison_axis_settings)

comparison = Line2D(average_equilateral_gr["k"], abs((average_equilateral_gr["B"] - average_equilateral_newton["B"])/average_equilateral_newton["B"]), color="black", label="Difference")
average_equilateral_fig.axes[1].add_line(comparison)
average_equilateral_fig.axes[1].autoscale_view()
leg2 = average_equilateral_fig.axes[1].legend(handles=[comparison], loc="lower right")

average_equilateral_fig.fig.savefig(f"../temp_figures/average_equilateral.png")
plt.close(average_equilateral_fig.fig)


average_squeezed_fig = CustomFigure()
settings = {
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"k $[h/Mpc]$",
            "ylabel": r"B(k,k,$\theta=19\pi/20$) $[h/Mpc]^3$",
            "title": f"Average Squeezed Bispectra of 'phi' at redshift z=1"
        }
average_squeezed_fig.set_settings(settings)
lines = []
lines.append(Line2D(average_squeezed_gr["k"], abs(average_squeezed_gr["B"]), color="blue", label="GR"))
lines.append(Line2D(average_squeezed_newton["k"], abs(average_squeezed_newton["B"]), color="red", label="Newton"))

for line in lines:
    average_squeezed_fig.ax.add_line(line)

leg1 = average_squeezed_fig.ax.legend(handles=lines, loc="upper right")

average_squeezed_fig.ax.add_artist(leg1)
average_squeezed_fig.ax.autoscale_view()

comparison_axis_settings = {
        # "xscale": "log",
        "yscale": "log",
        "ylabel": r"|GR-Newton|",
    }
ax2 = average_squeezed_fig.gen_twinx(comparison_axis_settings)

comparison = Line2D(average_squeezed_gr["k"], abs((average_squeezed_gr["B"] - average_squeezed_newton["B"])/average_squeezed_newton["B"]), color="black", label="Difference")
average_squeezed_fig.axes[1].add_line(comparison)
average_squeezed_fig.axes[1].autoscale_view()
leg2 = average_squeezed_fig.axes[1].legend(handles=[comparison], loc="lower right")

average_squeezed_fig.fig.savefig(f"../temp_figures/average_squeezed.png")
plt.close(average_squeezed_fig.fig)

plt.show()


# class AddBispectraComponents:
#     def __init__(self, return_pd:bool=False) -> None:
#         self.return_pd = return_pd

    # def add