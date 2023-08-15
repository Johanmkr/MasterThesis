import numpy as np
import matplotlib.pyplot as plt
import powerspectra as ps
import cambPK as caPk
import classPK as clPk

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
        self.cambObj = None
        self.classObj = None

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

    def init_camb(self) -> None:
        """
            Initialise the CAMB power spectra.
        """
        if isinstance(self.cambObj, caPk.CambSpectra):
            pass
        else:
            self.cambObj = caPk.CambSpectra()

    def init_class(self) -> None:
        """
            Initialise the CLASS power spectra.
        """
        if isinstance(self.classObj, clPk.ClassSpectra):
            pass
        else:
            self.classObj = clPk.ClassSpectra()

    def compare_camb(self, redshift:float=0.0, kmin:int=0, kmax:int=1000, save:bool=False) -> None:
        """
            Plot the power spectra for both GR and Newton.
            Args:
                redshift (float/int): The redshift of the power spectrum to plot.
                kmin (int): The minimum k to plot.
                kmax (int): The maximum k to plot.
                save (bool): Whether to save the plot or not.
        """

        # Initialise the CAMB and CLASS object
        self.init_camb()
        self.init_class()

        # Get the power spectra
        gr_spectrum = self.grPS.get_power_spectrum("delta", redshift)
        newton_spectrum = self.newtonPS.get_power_spectrum("delta", redshift)
        gr_spectrum_pk = gr_spectrum["pk"] / gr_spectrum["k"]**3 * 2*np.pi**2
        newton_spectrum_pk = newton_spectrum["pk"] / newton_spectrum["k"]**3 * (2*np.pi)**2
        camb_spectrum = self.cambObj(redshift)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(gr_spectrum["k"], gr_spectrum_pk, label="GR", color="blue")
        ax.plot(newton_spectrum["k"], gr_spectrum_pk, label="Newton", ls="--", color="red")
        ax.plot(*camb_spectrum, label="CAMB", ls=":", color="green")
        ax.plot(*self.classObj(), label="CLASS", ls=":", color="orange")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel(r"P(k) [Mpc/h]$^3$")
        ax.set_title(f"Power spectrum of 'delta' at redshift z={redshift}, for seed {self.grPS.seed}")
        # ax.set_xlim(kmin, kmax)
        ax.legend()

        # # New axis right below the first one for the ratio
        # ax2 = ax.twinx()
        # diff = np.abs(gr_spectrum["pk"] / newton_spectrum["pk"])
        # ax2.plot(gr_spectrum["k"], diff, color="black", ls="--", label="diff")
        # ax2.set_xscale("log")
        # ax2.set_xscale("log")
        # ax2.set_xlabel("k")
        # ax2.set_ylabel("P(k) GR - P(k) Newton")
        # # ax2.set_xlim(kmin, kmax)
        # ax2.set_ylim(0, diff.max()+diff.max()*0.1)
        # ax2.grid()
        # ax2.legend()
    

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
        redshift = 0.0

    path = datapath + f"seed{seed_nr:04d}/"
    obj = PlotPowerSpectra(path)
    # obj.plot_ps(pk_type=pktype, redshift=redshift)
    obj.compare_camb(redshift=redshift)