""" 
    Module to generate power spectra plots. Need to make the following spectra:

    1) CLASS and CAMB linear spectra (difference between Newtonian and Synchronous gauge)

    2) 9x9 plot showing 3 different seeds at three different redshifts with CAMB and CLASS spectra overplotted. Showing both GR and Newton matter power spectra. 

    3) 1 plot with most seeds plotted in background and average (with variance) plotted in foreground. At redshift 1, matter power spectrum.

    4) 9x9 plot showing 3 different seeds at three different redshifts for the potential power spectrum. Both spectra from gevolution and from cube directly.

    5) 1 plot with most seeds plotted in background and average (with variance) plotted in foreground. At redshift 1, potential power spectrum. Both spectra from gevolution and from cube directly. 
"""


import numpy as np
import matplotlib.pyplot as plt
from plotPS import AddPowerSpectraComponents
import paths
import pandas as pd
from figure import CustomFigure, SaveShow

from IPython import embed

####
# General stuff
####

boxsize = 5120 #Mpc
ngrid = 256 #px
resolution = boxsize/ngrid #Mpc/px
kN = np.pi / resolution 
kF = 2*np.pi/boxsize

three_seeds:list = [1001, 1045, 1956]
three_redshifts:list = [0, 1, 5]

power_spectrum_settings:dict = {
    "xscale": "log", 
    "yscale": "log",
    "xlim": (kF, kN),
    "ylim": (1e0, 1e5),
}



def matter_power_spectra():
    cfig = CustomFigure(ncols=3, nrows=3, figsize=(15, 15), sharex=True, sharey=True, gridspec_kw={"hspace": 0, "wspace": 0})
    # embed()
    for i, seed in enumerate(three_seeds):
        for j, redshift in enumerate(three_redshifts):

            # Get the index and axis stuff right
            idx_1d = i*3 + j
            ax = cfig.axes[j][i]
            ax.set(**power_spectrum_settings)
            adder = AddPowerSpectraComponents(paths.get_dir_with_seed(seed))
            local_lines = []

            # Add power spectra 
            # local_lines.extend(adder.add_gr_newton_gev("delta", redshift, color="blue"))
            local_lines.append(adder.add_newton_gev("delta", redshift, color="red"))
            local_lines.append(adder.add_gr_gev("delta", redshift, color="blue"))
            
            # Add CAMB and CLASS
            local_lines.append(adder.add_CAMB_spectrum(redshift, color="green"))
            local_lines.append(adder.add_CLASS_spectrum(redshift, gauge="newtonian", color="orange"))


            # Add lines to plot
            cfig.lines[idx_1d][f"ax{idx_1d}"] = local_lines
            for line in local_lines:
                ax.add_line(line)
            ax.autoscale_view()

            # Add labels
            if i==0:
                ax.set_ylabel(r"$P(k)\;[Mpc/h]^{-3}$")
            if j==2:
                ax.set_xlabel(r"$k\;[h/Mpc]$")
            
            # Add titles
            if j==0:
                ax.set_title(f"seed {seed}")
            if i==2:
                twin = ax.twinx()
                twin.yaxis.set_ticks([])
                twin.set_ylabel(f"z={redshift}", fontdict={"family": ax.title.get_fontfamily()[0], "size": ax.title.get_fontsize(), "weight": ax.title.get_fontweight()})
                twin.yaxis.set_label_coords(1.05, 0.5)
                # twin.yaxis.set_rotate_label(True)
                twin.yaxis.label.set_rotation(270)
            
    cfig.fig.suptitle(r"Matter powerspectrum $P_{\delta}^{\mathrm{gev}}(k)$")
    
    SaveShow(cfig, save_name="nine_matter_power_spectra", save=True, show=True, tight_layout=True)

def average_power_spectrum(redshift:int=1, pk_type:str="delta"):
    cfig = CustomFigure(ncols=1, nrows=1, figsize=(15,15))
    ax = cfig.axes[0]
    ax.set(ylabel=r"$P(k)\;[Mpc/h]^{-3}$", xlabel=r"$k\;[h/Mpc]$", **power_spectrum_settings)
    adder = AddPowerSpectraComponents(paths.get_dir_with_seed(1001))

    # Set seeds
    seed_range = np.arange(0,2000,20)
    averages = adder.add_averages(pk_type, redshift, seed_range, keep_background_lines=True)

    # Add average lines
    ax.add_line(averages[1])
    ax.add_line(averages[0])


    # Add background lines
    for lines in zip(averages[2], averages[3]):
        ax.add_line(lines[0])
        ax.add_line(lines[1])


    # Add labels


    # Add titles
    cfig.fig.suptitle(r"Average matter powerspectrum $P_{\delta}^{\mathrm{gev}}(k)$ for $z=1$ ")

    SaveShow(cfig, save_name="average_matter_power_spectrum", save=True, show=True, tight_layout=True)
    


if __name__=="__main__":
    matter_power_spectra()
    average_power_spectrum()