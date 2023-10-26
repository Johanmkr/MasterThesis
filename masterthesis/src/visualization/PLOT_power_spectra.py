""" 
    Module to generate power spectra plots. Need to make the following spectra:

    1) CLASS and CAMB linear spectra (difference between Newtonian and Synchronous gauge)

    2) 9x9 plot showing 3 different seeds at three different redshifts with CAMB and CLASS spectra overplotted. Showing both GR and Newton matter power spectra. 

    3) 1 plot with most seeds plotted in background and average (with variance) plotted in foreground. At redshift 1, matter power spectrum.

    4) 9x9 plot showing 3 different seeds at three different redshifts for the potential power spectrum. Both spectra from gevolution and from cube directly.

    5) 1 plot with most seeds plotted in background and average (with variance) plotted in foreground. At redshift 1, potential power spectrum. Both spectra from gevolution and from cube directly. 
"""

# Global imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Local imports
# from .plotPS import AddPowerSpectraComponents
from ..utils import paths
from ..utils.figure import CustomFigure, SaveShow
from ..features import classPK, avg_powerspectra

# Temporary imports
from IPython import embed

####
# General stuff
####

boxsize = 5120  # Mpc
ngrid = 256  # px
resolution = boxsize / ngrid  # Mpc/px
kN = np.pi / resolution
kF = 2 * np.pi / boxsize

three_seeds: list = [1001, 1045, 1956]
three_redshifts: list = [0, 1, 5]

matter_power_spectrum_settings: dict = {
    "xscale": "log",
    "yscale": "log",
    "xlim": (kF, kN),
    "ylim": (1e0, 1e5),
}

potential_power_spectrum_settings: dict = {
    "xscale": "log",
    "yscale": "log",
    "xlim": (kF, kN),
    "ylim": (1e-8, 5e1),
}


def average_matter_power_spectra():
    averageFig = CustomFigure(
        ncols=2,
        nrows=3,
        figsize=(10, 15),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0, "wspace": 0},
    )
    for i, z in enumerate(three_redshifts):
        # Calculate and find stuff
        # Class spectra
        pk_synchronous = classPK.ClassSpectra(z, "synchronous")
        pk_newtonian = classPK.ClassSpectra(z, "newtonian")
        synch_frame = pk_synchronous.d_tot_pk
        newt_frame = pk_newtonian.d_tot_pk

        # Average matter spectra
        avg_spectra = avg_powerspectra.AVG_powerspectra(z=z)
        avg_newton = avg_spectra.get_mean_std(gravity="newton")
        avg_gr = avg_spectra.get_mean_std(gravity="gr")

        # Generate plot for newtonian case
        ax1 = averageFig.axes[i][0]
        ax1.set(**matter_power_spectrum_settings)
        ax1.set_ylabel(r"$P(k)\;[Mpc/h]^{-3}$")
        # ax1.set_title(f"z={z}")
        # Set lines and stuff
        mean_newton = ax1.loglog(
            avg_newton["k"], avg_newton["mean"], color="red", label="Newtonian"
        )
        std_newton = ax1.fill_between(
            avg_newton["k"],
            avg_newton["mean"] - avg_newton["std"],
            avg_newton["mean"] + avg_newton["std"],
            alpha=0.2,
            color="red",
        )

        # ax1.legend()

        # Generate plot for GR case
        ax2 = averageFig.axes[i][1]
        ax2.set(**matter_power_spectrum_settings)
        # ax2.set_title(f"z={z}")
        # ax2.set_ylabel(r"$P(k)\;[Mpc/h]^{-3}$")
        ax2twin = ax2.twinx()
        # remove tick from ax2twin
        ax2twin.yaxis.set_ticks([])
        ax2twin.set_ylabel(
            f"z={z}",
            fontdict={
                "family": ax2.title.get_fontfamily()[0],
                "size": ax2.title.get_fontsize(),
                "weight": ax2.title.get_fontweight(),
            },
        )

        if i == 0:
            ax1.set_title("Newtonian")
            ax2.set_title("GR")

        if i == 2:
            ax1.set_xlabel(r"$k\;[h/Mpc]$")
            ax2.set_xlabel(r"$k\;[h/Mpc]$")

        # Set lines and stuff
        mean_gr = ax2.loglog(avg_gr["k"], avg_gr["mean"], color="blue", label="GR")
        std_gr = ax2.fill_between(
            avg_gr["k"],
            avg_gr["mean"] - avg_gr["std"],
            avg_gr["mean"] + avg_gr["std"],
            alpha=0.2,
            color="blue",
        )

        # ax2.legend()

        ### TODO IS THIS CORRECT GUAUGE
        analytical_newton = ax2.loglog(
            newt_frame["k"], newt_frame["pk"], color="black", ls="--", label="CLASS"
        )
        analytical_gr = ax1.loglog(
            synch_frame["k"], synch_frame["pk"], color="black", ls="--", label="CLASS"
        )
    averageFig.fig.tight_layout()
    averageFig.fig.suptitle(
        r"Average matter powerspectrum $P_{\delta}^{\mathrm{gev}}(k)$"
    )
    SaveShow(
        averageFig,
        save_name="average_matter_power_spectra",
        save=True,
        show=True,
        tight_layout=True,
    )


def average_potential_power_spectra():
    averageFig = CustomFigure(
        ncols=2,
        nrows=3,
        figsize=(10, 15),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0, "wspace": 0},
    )
    for i, z in enumerate(three_redshifts):
        # Calculate and find stuff
        # Class spectra
        pk_synchronous = classPK.ClassSpectra(z, "synchronous")
        pk_newtonian = classPK.ClassSpectra(z, "newtonian")
        synch_frame = pk_synchronous.phi_pk
        newt_frame = pk_newtonian.phi_pk

        # Average matter spectra
        avg_spectra = avg_powerspectra.AVG_powerspectra(pk_type="phi", z=z)
        avg_newton = avg_spectra.get_mean_std(gravity="newton")
        avg_gr = avg_spectra.get_mean_std(gravity="gr")

        # Generate plot for newtonian case
        ax1 = averageFig.axes[i][0]
        ax1.set(**potential_power_spectrum_settings)
        ax1.set_ylabel(r"$P(k)\;[Mpc/h]^{-3}$")
        # ax1.set_title(f"z={z}")
        # Set lines and stuff
        mean_newton = ax1.loglog(
            avg_newton["k"], avg_newton["mean"], color="red", label="Newtonian"
        )
        std_newton = ax1.fill_between(
            avg_newton["k"],
            avg_newton["mean"] - avg_newton["std"],
            avg_newton["mean"] + avg_newton["std"],
            alpha=0.2,
            color="red",
        )

        # ax1.legend()

        # Generate plot for GR case
        ax2 = averageFig.axes[i][1]
        ax2.set(**potential_power_spectrum_settings)
        # ax2.set_title(f"z={z}")
        # ax2.set_ylabel(r"$P(k)\;[Mpc/h]^{-3}$")
        ax2twin = ax2.twinx()
        # remove tick from ax2twin
        ax2twin.yaxis.set_ticks([])
        ax2twin.set_ylabel(
            f"z={z}",
            fontdict={
                "family": ax2.title.get_fontfamily()[0],
                "size": ax2.title.get_fontsize(),
                "weight": ax2.title.get_fontweight(),
            },
        )

        if i == 0:
            ax1.set_title("Newtonian")
            ax2.set_title("GR")

        if i == 2:
            ax1.set_xlabel(r"$k\;[h/Mpc]$")
            ax2.set_xlabel(r"$k\;[h/Mpc]$")

        # Set lines and stuff
        mean_gr = ax2.loglog(avg_gr["k"], avg_gr["mean"], color="blue", label="GR")
        std_gr = ax2.fill_between(
            avg_gr["k"],
            avg_gr["mean"] - avg_gr["std"],
            avg_gr["mean"] + avg_gr["std"],
            alpha=0.2,
            color="blue",
        )

        # ax2.legend()

        ### TODO IS THIS CORRECT GUAUGE
        analytical_newton = ax2.loglog(
            newt_frame["k"], newt_frame["pk"], color="black", ls="--", label="CLASS"
        )
        analytical_gr = ax1.loglog(
            synch_frame["k"], synch_frame["pk"], color="black", ls="--", label="CLASS"
        )
    averageFig.fig.tight_layout()
    averageFig.fig.suptitle(
        r"Average potential powerspectrum $P_{\Phi}^{\mathrm{gev}}(k)$"
    )
    SaveShow(
        averageFig,
        save_name="average_potential_power_spectra",
        save=True,
        show=True,
        tight_layout=True,
    )


# def matter_power_spectra():
#     cfig = CustomFigure(
#         ncols=3,
#         nrows=3,
#         figsize=(15, 15),
#         sharex=True,
#         sharey=True,
#         gridspec_kw={"hspace": 0, "wspace": 0},
#     )
#     # embed()
#     for i, seed in enumerate(three_seeds):
#         for j, redshift in enumerate(three_redshifts):
#             # Get the index and axis stuff right
#             idx_1d = i * 3 + j
#             ax = cfig.axes[j][i]
#             ax.set(**power_spectrum_settings)
#             adder = AddPowerSpectraComponents(paths.get_dir_with_seed(seed))
#             local_lines = []

#             # Add power spectra
#             # local_lines.extend(adder.add_gr_newton_gev("delta", redshift, color="blue"))
#             local_lines.append(adder.add_newton_gev("delta", redshift, color="red"))
#             local_lines.append(adder.add_gr_gev("delta", redshift, color="blue"))

#             # Add CAMB and CLASS
#             local_lines.append(adder.add_CAMB_spectrum(redshift, color="green"))
#             local_lines.append(
#                 adder.add_CLASS_spectrum(redshift, gauge="newtonian", color="orange")
#             )

#             # Add lines to plot
#             cfig.lines[idx_1d][f"ax{idx_1d}"] = local_lines
#             for line in local_lines:
#                 ax.add_line(line)
#             ax.autoscale_view()

#             # Add labels
#             if i == 0:
#                 ax.set_ylabel(r"$P(k)\;[Mpc/h]^{-3}$")
#             if j == 2:
#                 ax.set_xlabel(r"$k\;[h/Mpc]$")

#             # Add titles
#             if j == 0:
#                 ax.set_title(f"seed {seed}")
#             if i == 2:
#                 twin = ax.twinx()
#                 twin.yaxis.set_ticks([])
#                 twin.set_ylabel(
#                     f"z={redshift}",
#                     fontdict={
#                         "family": ax.title.get_fontfamily()[0],
#                         "size": ax.title.get_fontsize(),
#                         "weight": ax.title.get_fontweight(),
#                     },
#                 )
#                 twin.yaxis.set_label_coords(1.05, 0.5)
#                 # twin.yaxis.set_rotate_label(True)
#                 twin.yaxis.label.set_rotation(270)

#     cfig.fig.suptitle(r"Matter powerspectrum $P_{\delta}^{\mathrm{gev}}(k)$")

#     SaveShow(
#         cfig,
#         save_name="nine_matter_power_spectra",
#         save=True,
#         show=True,
#         tight_layout=True,
#     )


# def average_power_spectrum(redshift: int = 1, pk_type: str = "delta"):
#     cfig = CustomFigure(ncols=1, nrows=1, figsize=(15, 15))
#     ax = cfig.axes[0]
#     ax.set(
#         ylabel=r"$P(k)\;[Mpc/h]^{-3}$",
#         xlabel=r"$k\;[h/Mpc]$",
#         **power_spectrum_settings,
#     )
#     adder = AddPowerSpectraComponents(paths.get_dir_with_seed(1001))

#     # Set seeds
#     seed_range = np.arange(0, 2000, 20)
#     averages = adder.add_averages(
#         pk_type, redshift, seed_range, keep_background_lines=True
#     )

#     # Add average lines
#     ax.add_line(averages[1])
#     ax.add_line(averages[0])

#     # Add background lines
#     for lines in zip(averages[2], averages[3]):
#         ax.add_line(lines[0])
#         ax.add_line(lines[1])

#     # Add labels

#     # Add titles
#     cfig.fig.suptitle(
#         r"Average matter powerspectrum $P_{\delta}^{\mathrm{gev}}(k)$ for $z=1$ "
#     )

#     SaveShow(
#         cfig,
#         save_name="average_matter_power_spectrum",
#         save=True,
#         show=True,
#         tight_layout=True,
#     )


if __name__ == "__main__":
    matter_power_spectra()
    average_power_spectrum()
