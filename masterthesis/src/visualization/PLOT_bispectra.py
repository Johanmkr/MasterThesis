"""
    DESCRIPTION OF MODULE:

    
"""

# Global imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Local imports
from ..utils import paths
from ..features import avg_bispectra, bispectrum, analytical_bispectrum
from ..utils.figure import CustomFigure, SaveShow

# Temporary imports
from IPython import embed

redshifts = [0, 1, 10]


def analytical_bispectra():
    plot_settings = {}
    # Initialise the figure
    analFig = CustomFigure(
        nrows=3,
        ncols=1,
        figsize=(15, 12),
        sharex=True,
        gridspec_kw={"hspace": 0.0, "wspace": 0.0},
    )

    angles = [np.pi / 20, np.pi / 3, 2 * np.pi / 3, 19 * np.pi / 20]
    angle_names = [r"$\pi/20$", r"$\pi/3$", r"$2\pi/3$", r"$19\pi/20$"]
    k_range = np.geomspace(1e-5, 1e-1, 1000)

    for i, z in enumerate(redshifts):
        ax = analFig.axes[i]
        bs = analytical_bispectrum.AnalyticalBispectrum(k_range, z)
        for j, theta in enumerate(angles):
            ax.loglog(
                k_range, bs.get_custom_bispectrum(k_range, theta), label=angle_names[j]
            )
        ax2 = ax.twinx()
        ax2.set_ylabel(
            f"z={z}",
            fontdict={
                "family": ax.title.get_fontfamily()[0],
                "size": ax.title.get_fontsize(),
                "weight": ax.title.get_fontweight(),
            },
        )
        ax2.set_yticks([])
        if i == 0:
            ax.legend()
        if i == 2:
            ax.set_xlabel(r"$k\;[h/Mpc]$")

        ### TODO fill in units of bispectra
        ax.set_ylabel(r"$B(k, k, \theta)$")

    # Set the title and legend
    analFig.fig.suptitle(f"Analytical bispectra at different redshifts")

    # Save and show the figure
    SaveShow(
        analFig,
        save_name="analytical_bispectra",
        save=True,
        show=True,
        tight_layout=True,
    )


def binning_example(seed: float = 0, gravity: str = "gr", redshift: float = 1.0):
    binFig = CustomFigure(
        ncols=3,
        nrows=3,
        figsize=(15, 15),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.0, "wspace": 0.0},
    )

    # Define the number of rebins
    rebins = [0, 1, 2]
    stride = 5
    binnings = ["Original", "Rebinned once", "Rebinned twice"]
    for j, seed in enumerate([0, 1000, 1950]):
        test = bispectrum.CubeBispectrum(seed, gravity, redshift, initial_rebin=False)
        for i, rebin_count in enumerate(rebins):
            ax = binFig.axes[j, i]

            # Plot the equilateral bispectrum
            ax.loglog(
                test.B_equilateral["k"],
                test.B_equilateral["B"],
                color="orange",
                label="Equilateral",
            )
            ax.loglog(
                test.B_squeezed["k"],
                test.B_squeezed["B"],
                color="purple",
                label="Squeezed",
            )
            # binFig.axes[row, col].legend()
            # binFig.axes[row, col].set_title(
            #     f"Bispectrum (rebinned {rebin_count} times) with stride: {stride}"
            # )
            if rebin_count < 3:
                try:
                    test.rebin(bin_stride=stride)  # Apply rebinning
                except ValueError:
                    print("ValueError, cannot rebin anymore with the chosen stride")

            if j == 0:
                ax.set_title(binnings[i])
            if j == 2:
                ax.set_xlabel(r"$k\;[h/Mpc]$")
            if i == 0:
                ax.set_ylabel(r"$B(k, k, \theta)\;[Mpc/h]^{-3}$")
            if i == 2:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(
                    f"Seed: {seed}",
                    fontdict={
                        "family": ax.title.get_fontfamily()[0],
                        "size": ax.title.get_fontsize(),
                        "weight": ax.title.get_fontweight(),
                    },
                )
            if i == 2 and j == 0:
                ax.legend()

    # Add spacing between subplots
    binFig.fig.suptitle(
        f"Binning example at redshift: {redshift} and gravity: {gravity}"
    )
    SaveShow(
        binFig, save_name="binning_example", save=True, show=True, tight_layout=True
    )


def average_bispectra():
    avgFig = CustomFigure(
        ncols=3,
        nrows=2,
        figsize=(15, 15),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.2, "wspace": 0.0},
    )

    for i, z in enumerate(redshifts):
        avg_bispectrum = avg_bispectra.AVG_bispectrum(z=z)
        for j, b_type in enumerate(["equilateral", "squeezed"]):
            statistic_gr = avg_bispectrum.get_mean_std(gravity="gr", type=b_type)
            statistic_newton = avg_bispectrum.get_mean_std(
                gravity="newton", type=b_type
            )
            ax = avgFig.axes[j, i]

            # Mean and std of GR
            ax.loglog(
                statistic_gr.index, statistic_gr["mean"], label=f"GR", color="blue"
            )
            ax.fill_between(
                statistic_gr.index,
                statistic_gr["mean"] - statistic_gr["std"],
                statistic_gr["mean"] + statistic_gr["std"],
                alpha=0.2,
                color="blue",
            )
            # Mean and std of Newton
            ax.loglog(
                statistic_newton.index,
                statistic_newton["mean"],
                label=f"Newton",
                color="red",
            )
            ax.fill_between(
                statistic_newton.index,
                statistic_newton["mean"] - statistic_newton["std"],
                statistic_newton["mean"] + statistic_newton["std"],
                alpha=0.2,
                color="red",
            )

            # Add analytical bispectrum
            k_range_anal = np.geomspace(
                statistic_newton.index[0], statistic_newton.index[-1], 1000
            )
            analytical = analytical_bispectrum.AnalyticalBispectrum(k_range_anal, z)

            if b_type == "equilateral":
                ax.loglog(
                    k_range_anal,
                    analytical.B_equilateral,
                    label="Analytical",
                    color="black",
                    ls="--",
                )
            if b_type == "squeezed":
                ax.loglog(
                    k_range_anal,
                    analytical.B_squeezed,
                    label="Analytical",
                    color="black",
                    ls="--",
                )

            # Plot difference between GR and Newton

            ax2 = ax.inset_axes([0, 1, 1, 0.5])
            ax2.semilogx(
                statistic_newton.index,
                np.abs((statistic_gr["mean"] - statistic_newton["mean"]))
                / statistic_newton["mean"],
                label=r"$|(B_{GR} - B_{Newton})/B_{Newton}|$",
                color="green",
            )

            # Rescale axis to fit in figure
            pos1 = ax.get_position()  # Get the position of the main subplot
            pos2 = [
                pos1.x0,
                pos1.y0,
                pos1.width,
                pos1.height * 0.8,
            ]  # Adjust the height
            ax.set_position(pos2)

            ax2.set_ylim(0, 0.035)

            # Add appropriate labels
            if i != 0:
                ax2.set_yticks([])
            ax2.set_xticks([])

            if i == 0:
                ax.set_ylabel(r"$B(k, k, \theta) [Mpc/h]^{-3}$")
            if j == 1:
                ax.set_xlabel(r"$k\;[h/Mpc]$")
            if i == 2 and j == 0:
                ax.legend()
            if i == 2 and j == 1:
                ax2.legend(loc="upper left")

            if i == 2:
                labelax = ax.twinx()
                labelax.set_position(pos2)
                labelax.set_yticks([])
                labelax.set_ylabel(
                    f"{b_type.capitalize()}",
                    fontdict={
                        "family": ax.title.get_fontfamily()[0],
                        "size": ax.title.get_fontsize(),
                        "weight": ax.title.get_fontweight(),
                    },
                )
            if j == 0:
                ax.set_title(f"z={z}")
    SaveShow(
        avgFig, save_name="average_bispectra", save=True, show=True, tight_layout=True
    )


if __name__ == "__main__":
    pass
