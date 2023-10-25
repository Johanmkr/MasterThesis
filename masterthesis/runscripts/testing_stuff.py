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
from src.features import analytical_bispectrum, bispectrum, avg_bispectra


from IPython import embed


def testing_analytic_bispectrum():
    test = analytical_bispectrum.AnalyticalBispectrum(
        k_range=np.geomspace(1e-5, 1e-1, 1000), z=0.0
    )
    theta_range = [np.pi / 3, np.pi / 2, 2 * np.pi / 3, 19 * np.pi / 20]
    for theta in theta_range:
        plt.loglog(
            test.k_range, test.get_custom_bispectrum(test.k_range, theta), label=theta
        )
    # plt.loglog(test.k_range, test.B_equilateral, label="equilateral")
    # plt.loglog(test.k_range, test.B_squeezed, label="squeezed")
    plt.title("Analytical bispectrum at z=1.0")
    plt.legend()
    plt.show()


def testing_pre_computed_bispectrum(
    seed: float = 0, gravity: str = "newton", redshift: float = 1.0
):
    test = bispectrum.CubeBispectrum(seed, gravity, redshift, initial_rebin=False)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    # Define the number of rebins
    rebins = [0, 1, 2, 3]
    stride = 5

    for i, rebin_count in enumerate(rebins):
        row = i // 2  # Calculate the row index (0 or 1)
        col = i % 2  # Calculate the column index (0 or 1)

        # Plot the equilateral bispectrum
        axs[row, col].loglog(
            test.B_equilateral["k"], test.B_equilateral["B"], label="equilateral"
        )
        axs[row, col].loglog(
            test.B_squeezed["k"], test.B_squeezed["B"], label="squeezed"
        )
        axs[row, col].legend()
        axs[row, col].set_title(
            f"Bispectrum (rebinned {rebin_count} times) with stride: {stride}"
        )
        if rebin_count < 3:
            try:
                test.rebin(bin_stride=stride)  # Apply rebinning
            except ValueError:
                print("ValueError, cannot rebin anymore with the chosen stride")

    # Add spacing between subplots
    plt.tight_layout()
    plt.show()


def testing_average_bispectrum(redshift: float = 1.0):
    # avg_bispectrum = avg_bispectra.AVG_bispectrum(z=redshift)
    # equilateral_statistic = avg_bispectrum.get_mean_std(
    #     gravity="gr", type="equilateral"
    # )
    # squeezed_statistic = avg_bispectrum.get_mean_std(type="squeezed")

    # plt.loglog(
    #     equilateral_statistic.index,
    #     equilateral_statistic["mean"],
    #     label="equilateral",
    #     color="green",
    # )
    # plt.loglog(
    #     squeezed_statistic.index,
    #     squeezed_statistic["mean"],
    #     label="squeezed",
    #     color="purple",
    # )

    # # Add standard deviations
    # plt.fill_between(
    #     equilateral_statistic.index,
    #     equilateral_statistic["mean"] - equilateral_statistic["std"],
    #     equilateral_statistic["mean"] + equilateral_statistic["std"],
    #     alpha=0.2,
    #     color="green",
    # )

    # plt.fill_between(
    #     squeezed_statistic.index,
    #     squeezed_statistic["mean"] - squeezed_statistic["std"],
    #     squeezed_statistic["mean"] + squeezed_statistic["std"],
    #     alpha=0.2,
    #     color="purple",
    # )

    # plt.legend()
    # plt.title(f"Average bispectrum at z={redshift}")
    # plt.show()
    redshifts = [0, 1]
    gravity_types = ["gr", "newton"]
    bispectrum_types = ["equilateral", "squeezed"]

    # Create a figure and a grid of subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    for i, bispectrum_type in enumerate(bispectrum_types):
        for j, redshift in enumerate(redshifts):
            ax = axs[j, i]  # Corresponding subplot
            avg_bispectrum = avg_bispectra.AVG_bispectrum(z=redshift)
            statistic_gr = avg_bispectrum.get_mean_std(
                gravity="gr", type=bispectrum_type
            )
            statistic_newton = avg_bispectrum.get_mean_std(
                gravity="newton", type=bispectrum_type
            )

            # Plot mean values
            ax.loglog(
                statistic_gr.index, statistic_gr["mean"], label="GR", color="blue"
            )

            # Add standard deviation as fill
            ax.fill_between(
                statistic_gr.index,
                statistic_gr["mean"] - statistic_gr["std"],
                statistic_gr["mean"] + statistic_gr["std"],
                alpha=0.2,
                color="blue",
            )

            ax.loglog(
                statistic_newton.index,
                statistic_newton["mean"],
                label="Newtonian",
                color="red",
            )

            ax.fill_between(
                statistic_newton.index,
                statistic_newton["mean"] - statistic_newton["std"],
                statistic_newton["mean"] + statistic_newton["std"],
                alpha=0.2,
                color="red",
            )

            # twin y axis and plot the relative difference between newton and gr
            ax2 = ax.twinx()
            ax2.semilogx(
                statistic_newton.index,
                np.abs((statistic_gr["mean"] - statistic_newton["mean"]))
                / statistic_newton["mean"],
                label="relative difference",
                color="green",
            )

            ax.legend()
            ax2.legend(loc="lower left")
            ax.set_title(f"{bispectrum_type} - z={redshift}")

    # Add spacing between subplots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    testing_analytic_bispectrum()
    testing_pre_computed_bispectrum(1050, "gr", 0.0)
    testing_average_bispectrum(0)
