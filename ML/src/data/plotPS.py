import numpy as np
import matplotlib.pyplot as plt
import powerspectra as ps

class PlotPowerSpectra:
    def __init__(self, data_dir:str) -> None:
        """
            Initialise the PowerSpectra objects for both GR and Newton.
            Args:
                data_dir (str): The directory containing the power spectra.
        """
        self.dataDir = data_dir
        self.grPS = ps.PowerSpectra(self.dataDir + "gr")
        self.newtonPS = ps.PowerSpectra(self.dataDir + "newton")

    def plot_ps(self, pk_type:str="delta", redshift:float=0., kmin:int=0, kmax:int=1000, save:bool=False) -> None:
        """
            Plot the power spectra for both GR and Newton.
            Args:
                pk_type (str): The type of power spectrum to plot. Can be ["deltacdm",      "deltaclass", "delta", "phi"] for GR and ["delta", "deltaclass", "phi"] for Newton. Defaults to "delta".
                redshift (float/int): The redshift of the power spectrum to plot. Defaults to 0.
                kmin (int): The minimum k to plot.
                kmax (int): The maximum k to plot.
                save (bool): Whether to save the plot or not.
        """

        # Get the power spectra
        gr_spectrum = self.grPS.get_power_spectrum(pk_type, redshift)
        newton_spectrum = self.newtonPS.get_power_spectrum(pk_type, redshift)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(gr_spectrum["k"], gr_spectrum["pk"], label="GR", color="blue")
        ax.plot(newton_spectrum["k"], gr_spectrum["pk"], label="Newton", ls="--", color="red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k")
        ax.set_ylabel("P(k)")
        ax.set_title(f"Power spectrum of '{pk_type}' at redshift z={redshift}, for seed {self.grPS.seed}")
        # ax.set_xlim(kmin, kmax)
        ax.legend()

        # New axis right below the first one for the ratio
        ax2 = ax.twinx()
        diff = np.abs(gr_spectrum["pk"] - newton_spectrum["pk"])
        ax2.plot(gr_spectrum["k"], diff, color="black", ls="--", label="diff")
        ax2.set_xscale("log")
        ax2.set_xscale("log")
        ax2.set_xlabel("k")
        ax2.set_ylabel("P(k) GR - P(k) Newton")
        # ax2.set_xlim(kmin, kmax)
        ax2.set_ylim(0, diff.max()+diff.max()*0.1)
        ax2.grid()
        ax2.legend()
    

        if save:
            plt.savefig(self.dataDir + "ps.png")
        else:
            plt.show()



if __name__=="__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    print("Plotting power spectra")
    if input("Enter info manually? [y/n]: ") in ["y", "yes", "Y", "Yes"]:
        seed_nr = int(input("Enter seed number: "))
        pktype = input("Enter power spectrum type (deltacdm(gr only), deltaclass, delta, phi): ")
        redshift = float(input("Enter redshift (100, 50, 20, 10, 6, 5, 4, 3, 2, 1, .9, .8, .7, .6, .5, .4, .3, .2, .1, .0): "))
    else:
        seed_nr = 0000
        pktype = "delta"
        redshift = 50

    path = datapath + f"seed{seed_nr:04d}/"
    obj = PlotPowerSpectra(path)
    obj.plot_ps(pk_type=pktype, redshift=redshift)