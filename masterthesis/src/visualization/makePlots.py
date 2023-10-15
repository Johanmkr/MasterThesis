"""
    DESCRIPTION OF MODULE:

    
"""

from . import plotPS as pps
import numpy as np

# This is bad code ahahaha but I dont care
if __name__ == "__main__":
    infostr = "What would you like to plot?\n"
    infostr += "1: Power spectrum directly from gevolution.\n"
    infostr += "2: Matter power spectrum compared with CAMB and CLASS.\n"
    infostr += "3: Power spectrum averaged over some seeds\n"
    infostr += "4: Potential power spectrum from both gevolution and cube itself\n"
    infostr += "Enter up to (4) numbers: "
    response = input(infostr)

    if "1" in response:
        plot_gev_ps = True
    else:
        plot_gev_ps = False
    if "2" in response:
        plot_matter_ps = True
    else:
        plot_matter_ps = False
    if "3" in response:
        plot_avg_ps = True
    else:
        plot_avg_ps = False
    if "4" in response:
        plot_pot_ps = True
    else:
        plot_pot_ps = False

    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"

    if input("Enter cube manually? ") in ["y", "yes", "Y"]:
        try:
            seed_nr = int(input("Enter seed [0000 - 1999]: "))
        except ValueError:
            seed_nr = 1234
            print(f"Using default seed {seed_nr}")
        try:
            pk_type = input(
                "Enter power spectrum type (deltacdm(gr only), deltaclass, delta, phi): "
            )
        except ValueError:
            pk_type = "delta" if not plot_pot_ps else "phi"
            print(f"Using default power spectrum type {pk_type}")
        try:
            redshift = int(input("Enter redshift [0, 1, 5, 10, 15, 20]: "))
        except ValueError:
            redshift = 1
            print(f"Using default redshift {redshift}")
    else:
        seed_nr = 1234
        pk_type = "delta" if not plot_pot_ps else "phi"
        redshift = 1

    obj = pps.PlotPowerSpectra(datapath + f"seed{seed_nr:04d}/")
    if plot_gev_ps:
        obj.plot_ps(pk_type=pk_type, redshift=redshift)
    if plot_matter_ps:
        obj.compare_camb_class(redshift=redshift)
    if plot_avg_ps:
        obj.plot_average_ps(
            pk_type=pk_type, redshift=redshift, seed_range=np.arange(2000, dtype=int)
        )
    if plot_pot_ps:
        obj.plot_cube_ps(redshift=redshift)

    # print(seed_nr)
