import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from IPython import embed
import os, sys
# Parent forlder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figure as fg
from bispectra import analytical_bispectrum as B_anal

# Global variables
boxsize = 5120  # Mpc/h
Ngrid = 256
resolution = boxsize / Ngrid
kF = 2 * np.pi / boxsize
kN = np.pi / resolution

avg_path = "/mn/stornext/d10/data/johanmkr/simulations/bispectra_analysis/average_bispectra/"
B_avg_path = lambda kind, A_s, gravity: avg_path + f"B_avg_{kind}_{A_s:.2e}_{gravity.lower()}.pkl"

z10gr = pd.read_pickle(B_avg_path("z10", 2.215e-9, "GR"))
z10newton = pd.read_pickle(B_avg_path("z10", 2.215e-9, "Newton"))
z1gr = pd.read_pickle(B_avg_path("z1", 2.215e-9, "GR"))
z1newton = pd.read_pickle(B_avg_path("z1", 2.215e-9, "Newton"))
scaled_gr = pd.read_pickle(B_avg_path("scaled", 2.215e-9, "GR"))
scaled_newton = pd.read_pickle(B_avg_path("scaled", 2.215e-9, "Newton"))

# remove first and last elements
z10gr = z10gr.iloc[1:-1]
z10newton = z10newton.iloc[1:-1]
z1gr = z1gr.iloc[1:-1]
z1newton = z1newton.iloc[1:-1]
scaled_gr = scaled_gr.iloc[1:-1]
scaled_newton = scaled_newton.iloc[1:-1]


# embed()

# def plot_equilateral_bispectrum():
#     # Set plots settings
#     plot_settings = dict(
#         xlabel=r"$k$ [h/Mpc]",
#         ylabel=r"$B_\Phi(k, \mu\approx0.5, t\approx1)$ [Mpc/h]$^6$",
#         title="Equilateral Bispectrum",
#         aspect="auto",
#         xscale="log",
#         yscale="log",
#         xlim=(1e-3, 3e-1),
#     )
#     # Make figure through wrapper
#     equi = fg.CustomFigure(nrows=1, ncols=1, figsize=(20, 12), settings=plot_settings)
#     plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.7, wspace=0.2, hspace=0.1)
    
#     # Plot z=10
#     equi.axes[0].plot(z10gr["k"], abs(z10gr["B_eq_avg"]), color="red", ls="dotted")
#     equi.axes[0].plot(z10newton["k"], abs(z10newton["B_eq_avg"]), color="blue", ls="dotted")
    
#     # Plot z=1
#     equi.axes[0].plot(z1gr["k"], abs(z1gr["B_eq_avg"]), color="red", ls="dashed")
#     equi.axes[0].plot(z1newton["k"], abs(z1newton["B_eq_avg"]), color="blue", ls="dashed")
    
#     # Plot difference
#     grdiff = equi.axes[0].plot(scaled_gr["k"], abs(scaled_gr["B_eq_avg"]), color="red", ls="solid", label="GR")
#     newtdiff = equi.axes[0].plot(scaled_newton["k"], abs(scaled_newton["B_eq_avg"]), ls="solid", color="blue", label="Newton")
    
    
#     #  Add axis above current axis to show relative difference
#     z10_diff = abs(z10gr["B_eq_avg"] - z10newton["B_eq_avg"]) / abs(z10newton["B_eq_avg"])
#     z1_diff = abs(z1gr["B_eq_avg"] - z1newton["B_eq_avg"]) / abs(z1newton["B_eq_avg"])
#     diff_diff = abs(scaled_gr["B_eq_avg"] - scaled_newton["B_eq_avg"]) / abs(scaled_newton["B_eq_avg"])
    
#     # tx = equi.axes[0].inset_axes([0.0, 0.8, 1, 0.5])
#     # tx.semilogx(z10gr["k"], z10_diff, color="k", ls="dotted")
#     # tx.semilogx(z1gr["k"], z1_diff, color="k", ls="dashed")
#     # tx.semilogx(scaled_gr["k"], diff_diff, color="k", ls="solid")
    
#     # # reshape origianl axis
#     # equi.axes[0].set_position([0.1, 0.1, 0.8, 0.6])
    
#     top_ax = equi.fig.add_axes([0.1, 0.8, 0.8, 0.2])  # [left, bottom, width, height] in figure coordinate

#     # Plot the data on the top panel.
#     top_ax.semilogx(z10gr["k"], z10_diff, color="k", ls="dotted")
#     top_ax.semilogx(z1gr["k"], z1_diff, color="k", ls="dashed")
#     top_ax.semilogx(scaled_gr["k"], diff_diff, color="k", ls="solid")
    
    
#     # Shade out invalid region where k is smaller than kF and larger than kN
#     equi.axes[0].axvspan(0, kF, alpha=0.5, color="grey")
#     equi.axes[0].axvspan(kN, 1e2, alpha=0.5, color="grey")
    
#     # Add legend
#     axleg = plt.legend([grdiff[0], newtdiff[0]], ["GR", "Newton"], loc="upper right")
#     plt.gca().add_artist(axleg)
#     # Create legend for linestyles
#     linestyles = ["dotted", "dashed", "solid"]
#     labels = [r"$\Phi_{z=10}$", r"$\Phi_{z=1}$", r"$\tilde{\Phi}$"]
#     ls_legends = equi.axes[0].legend(
#         [plt.Line2D((0, 0), (0, 0), color="k", linestyle=ls) for ls in linestyles],
#         labels,
#         loc="lower right",
#     )
    
    
    
    
    
    
    # Add newton-gr legend for the different colors
    # equi.axes[0].text(0.8, 0.95, "GR", color="red", transform=equi.axes[0].transAxes)
    # equi.axes[0].text(0.8, 0.92, "Newton", color="blue", transform=equi.axes[0].transAxes)
    # plt.show()
    
    
    
def plot_equilateral_bispectrum(spectrum="equilateral"):
    plot_settings = dict(
        xlabel=r"$k$ [h/Mpc]",
        ylabel=r"$B_\Phi(k, \mu, t)$ [Mpc/h]$^6$",
        aspect="auto",
        xscale="log",
        yscale="log",
        xlim=(1e-3, 2e-1),
        ylim=(1e-9, 1e6),
    )
    
    # Calculate the correct quantities
    if spectrum.lower() not in {"equilateral", "eq", "squeezed", "sq", "stretched", "st"}:
        raise ValueError("Invalid spectrum")
    if spectrum.lower() in {"equilateral", "eq"}:
        kind = "B_eq_avg"
        mu_val = 0.5
        t_val = 1
    elif spectrum.lower() in {"squeezed", "sq"}:
        kind = "B_sq_avg"
        mu_val = 0.99
        t_val = 0.99
    elif spectrum.lower() in {"stretched", "st"}:
        kind = "B_st_avg"
        mu_val = 0.99
        t_val = 0.5
    
    # Calculate the relative difference
    z10gr_data = (z10gr["k"], abs(z10gr[kind]))
    z10newton_data = (z10newton["k"], abs(z10newton[kind]))
    z1gr_data = (z1gr["k"], abs(z1gr[kind]))
    z1newton_data = (z1newton["k"], abs(z1newton[kind]))
    scaled_gr_data = (scaled_gr["k"], abs(scaled_gr[kind]))
    scaled_newton_data = (scaled_newton["k"], abs(scaled_newton[kind]))
    
    z10_diff = abs(z10gr[kind] - z10newton[kind]) / abs(z10newton[kind])
    z1_diff = abs(z1gr[kind] - z1newton[kind]) / abs(z1newton[kind])
    scaled_diff = abs(scaled_gr[kind] - scaled_newton[kind]) / abs(scaled_newton[kind])

    # Analytical bispectra
    k_anal = np.logspace(-3, 0, 100)
    z10anal = B_anal.get_Bk(k_anal, mu=mu_val, t=t_val, z=10, nl=True)
    z1anal = B_anal.get_Bk(k_anal, mu=mu_val, t=t_val, z=1, nl=True)
    
    
    fig = plt.figure(figsize=(15, 12))

    # Create a gridspec object with 2 rows.
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

    # Create the main ax, as the second subplot on the gridspec.
    main_ax = plt.subplot(gs[1])
    main_ax.set(**plot_settings)
  
    # Plot z=10
    main_ax.plot(*z10gr_data, color="red", ls="dotted")
    main_ax.plot(*z10newton_data, color="blue", ls="dotted")

    # Plot z=1
    main_ax.plot(*z1gr_data, color="red", ls="dashed")
    main_ax.plot(*z1newton_data, color="blue", ls="dashed")

    # Plot difference
    grdiff = main_ax.plot(*scaled_gr_data, color="red", ls="solid", label="GR")
    newtdiff = main_ax.plot(*scaled_newton_data, ls="solid", color="blue", label="Newton")
    
    
    # Plot analytical bispectra
    main_ax.plot(k_anal, z10anal, color="k", ls="dotted")
    main_ax.plot(k_anal, z1anal, color="k", ls="dashed")
    
    
    # Add legend
    axleg = plt.legend([grdiff[0], newtdiff[0]], ["GR", "Newton"], loc="upper right")
    plt.gca().add_artist(axleg)
    # Create legend for linestyles
    linestyles = ["dotted", "dashed", "solid"]
    labels = [r"$\Phi_{z=10}$", r"$\Phi_{z=1}$", r"$\tilde{\Phi}$"]



    # Create the top ax, for the first subplot on the gridspec.
    top_ax = plt.subplot(gs[0], sharex=main_ax)
    top_ax.semilogx(z10gr["k"], z10_diff, color="k", ls="dotted")
    top_ax.semilogx(z1gr["k"], z1_diff, color="k", ls="dashed")
    top_ax.semilogx(scaled_gr["k"], scaled_diff, color="k", ls="solid")
    

    ls_legends = top_ax.legend(
        [plt.Line2D((0, 0), (0, 0), color="k", linestyle=ls) for ls in linestyles],
        labels,
        loc="upper right",
    )
    # Disabling the x-tick labels for the top ax, as they are shared.
    plt.setp(top_ax.get_xticklabels(), visible=False)
    
    # Remove the largest tick from the y axis of the main panel
    main_ax.set_yticks(main_ax.get_yticks()[1:-3])
    
    
    main_ax.axvspan(0, kF, alpha=0.25, color="grey")
    main_ax.axvspan(kN, 1e2, alpha=0.25, color="grey")
    top_ax.axvspan(0, kF, alpha=0.25, color="grey")
    top_ax.axvspan(kN, 1e2, alpha=0.25, color="grey")
    
    title = f"{spectrum.capitalize()} Bispectrum (bottom) and Relative Difference (top)"
    fig.suptitle(title)
    
    # set text with mu and t values
    mu = r"$\mu\approx0.5$" if spectrum.lower() in {"equilateral", "eq"} else r"$\mu\approx1$"
    t = r"$t\approx0.5$" if spectrum.lower() in {"stretched", "st"} else r"$t\approx1$"
    text = f"{mu}\n{t}"
    main_ax.text(0.7, 0.85, text, color="k", transform=main_ax.transAxes, fontsize=20)
    
    

    # Adjust the space between the plots to be zero.  
    plt.subplots_adjust(hspace=0.0)
    
    savename = f"{spectrum}_bispectrum"
    fg.SaveShow(fig=fig, save_name=savename, save=True, show=True, tight_layout=True)
    
if __name__=="__main__":
    plot_equilateral_bispectrum(spectrum="equilateral")
    plot_equilateral_bispectrum(spectrum="squeezed")
    plot_equilateral_bispectrum(spectrum="stretched")
