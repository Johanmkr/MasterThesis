import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure import CustomFigure
import powerspectra as ps
import cambPK as caPk
import classPK as clPk
import cube
import pyliansPK
from typing import Union

from IPython import embed


# Local variabels used for testing
boxsize = 5120 #Mpcq
ngrid = 256 #px
resolution = boxsize/ngrid #Mpc/px
k_nyquist = np.pi / resolution 
k_boxsize = 2*np.pi/boxsize

# print(k_nyquist)
# print(k_max)



#TODO: Fix the general class below. 
#TODO: imporve the plt.savefig statements in all the plotting functions.
#TODO: Rewrite docstrings for all functions
class MakeFigures:
    def __init__(self, data_dir_without_seed:str) -> None:
        self.dataDirwoSeed = data_dir_without_seed

    def plot_ps(self, seeds:Union[int, tuple, list]=1234, pk_type:str="delta", redshift:float=0.0, save:bool=False) -> None:
        """
            Plot the power spectra for both GR and Newton.
            Args:
                pk_type (str): The type of power spectrum to plot. Can be ["deltacdm",      "deltaclass", "delta", "phi"] for GR and ["delta", "deltaclass", "phi"] for Newton. Defaults to "delta".
                redshift (float/int): The redshift of the power spectrum to plot. Defaults to 0.
                save (bool): Whether to save the plot or not.
        """

        figures = []

        if isinstance(seeds, int):
            seeds = [seeds]

        for seed in seeds:
            figure = CustomFigure()
            adder = AddPowerSpectraComponents(self.dataDirwoSeed + f"seed{seed:04d}/")
            settings = {
                "xscale": "log",
                "yscale": "log",
                "xlabel": r"k $[h/Mpc]$",
                "ylabel": r"P(k) $[h/Mpc]^3$",
                "title": f"Power spectrum of '{pk_type}' at redshift z={redshift:.1f}, for seed {seed:04d}"
            }

            figure.set_settings(settings)
            lines = []
            lines.append(adder.add_gr_gev(pk_type, redshift, color="blue"))
            lines.append(adder.add_newton_gev(pk_type, redshift, color="blue"))
            # lines.extend(adder.add_gr_newton_gev(pk_type, redshift, color="blue"))

            for line in lines:
                figure.ax.add_line(line)

            # Legend for GR and Newton
            leg1 = figure.ax.legend(handles=lines, loc="upper right")
            figure.ax.add_artist(leg1)

            figure.ax.autoscale_view()

            adder._add_gr_newton_legend(figure.ax)

        
            if save:
                plt.savefig(self.dataDir + "ps.png")
            else:
                plt.show()

    def compare_camb_class(self, seeds:Union[int, tuple, list]=1234, pk_type:str="delta", redshift:float=0.0, save:bool=False) -> None:
        """
            Compare the power spectra from CAMB and CLASS.
            Args:
                redshift (float/int): The redshift of the power spectrum to plot.
                save (bool): Whether to save the plot or not.   
        """

        comp_plot = CustomFigure()
        settings = {
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"k $[h/Mpc]$",
            "ylabel": r"P(k) $[h/Mpc]^3$",
            "title": f"Power spectrum of '{pk_type}' at redshift z={redshift:.1f}, for seed {self.grPS.seed:04d}"
        }

        comp_plot.set_settings(settings)
        self._add_gr_newton(comp_plot.ax, pk_type, redshift)
        # self._add_axis_limits(comp_plot.ax)
        self._add_camb_class(comp_plot.ax, redshift)
        self._add_limits(comp_plot.ax)
        comp_plot.ax.legend(loc="lower left")

        if save:
            plt.savefig(self.dataDir + "ps.png")
        else:
            plt.show()

    def plot_average_ps(self, seeds:Union[int, tuple, list]=1234, pk_type:str="delta", redshift:float=0.0, seed_range:Union[list, np.ndarray, tuple]=np.arange(10, dtype=int), save:bool=False) -> None:
        """
            Plot the average power spectrum for a given seed range.
            Args:
                pk_type (str): The type of power spectrum to plot. Can be ["deltacdm", "deltaclass", "delta", "phi"] for GR and ["delta", "deltaclass", "phi"] for Newton. Defaults to "delta".
                redshift (float/int): The redshift of the power spectrum to plot. Defaults to 0.
                seed_range (list, np.ndarray, tuple): The range of seeds to average over.
                save (bool): Whether to save the plot or not.
        """

        avg_plot = CustomFigure()

        settings = {
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"k $[h/Mpc]$",
            "ylabel": r"P(k) $[h/Mpc]^3$",
            "title": f"Average power spectrum of '{pk_type}' at redshift z={redshift:.1f}"
        }
        avg_plot.set_settings(settings)
        self._add_averages(avg_plot.ax, pk_type, redshift, seed_range)
        self._add_camb_class(avg_plot.ax, redshift)
        self._add_limits(avg_plot.ax)
        self._add_gr_newton(avg_plot.ax, pk_type, redshift, add_ratio=False)

        avg_plot.ax.legend(loc="lower left")

        if save:
            plt.savefig(self.dataDir + "ps.png")
        else:
            plt.show()

    def plot_cube_ps(self, seeds:Union[int, tuple, list]=1234, redshift:float=0.0, save:bool=False) -> None:
        """
            Plot the power spectra from the cube. This can only be found for phi as this is the only field that is saved.
            Args:
                redshift (float/int): The redshift of the power spectrum to plot.
                save (bool): Whether to save the plot or not.
        """
        cube_plot = CustomFigure()

        settings = {
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"k $[h/Mpc]$",
            "ylabel": r"P(k) $[h/Mpc]^3$",
            "title": f"Power spectrum of 'phi' at redshift z={redshift:.1f}, for seed {self.grPS.seed:04d}"
        }

        cube_plot.set_settings(settings)
        gr_newton = self._add_gr_newton(cube_plot.ax, "phi", redshift)
        self._add_cube_spectra(cube_plot.ax, redshift)
        limits = self._add_limits(cube_plot.ax)
        cube_plot.ax.legend(loc="lower left")

        if save:
            plt.savefig(self.dataDir + "ps.png")
        else:
            plt.show()


class AddPowerSpectraComponents:
    def __init__(self, data_dir_with_seed:str) -> None:
        """
            Initialise the PowerSpectra objects for both GR and Newton.
            Args:
                data_dir (str): The directory containing the power spectra (seed included)
        """
        self.dataDirwSeed = data_dir_with_seed
        self.grPS = ps.PowerSpectra(self.dataDirwSeed + "gr")
        self.newtonPS = ps.PowerSpectra(self.dataDirwSeed + "newton")

    def add_gr_gev(self, pk_type:str, redshift:float, **kwargs:dict) -> Line2D:
        gr_spectrum = self.grPS.get_power_spectrum(pk_type, redshift)
        gr_line = Line2D(gr_spectrum["k"], gr_spectrum["pk"], label="Gev", **kwargs)
        return gr_line

    def add_newton_gev(self, pk_type:str, redshift:float, **kwargs:dict) -> Line2D:
        newton_spectrum = self.newtonPS.get_power_spectrum(pk_type, redshift)
        newton_line = Line2D(newton_spectrum["k"], newton_spectrum["pk"], ls="--", **kwargs)
        return newton_line

    def add_gr_newton_gev(self, pk_type:str, redshift:float, **kwargs:dict) -> tuple:
        return self.add_gr_gev(pk_type, redshift, **kwargs), self.add_newton_gev(pk_type, redshift, **kwargs)
    
    def add_CAMB_spectrum(self, redshift:float, **kwargs:dict) -> Line2D:
        if not hasattr(self, "cambObj"):
            self._init_camb()
        camb_spectrum = self.cambObj(redshift)
        camb_line = Line2D(camb_spectrum[0], camb_spectrum[1], **kwargs)
        return camb_line
    
    def add_CLASS_spectrum(self, redshift:float, **kwargs:dict) -> Line2D:
        if not hasattr(self, "classObj"):
            self._init_class()
        class_spectrum = self.classObj(redshift)
        class_line = Line2D(class_spectrum[0], class_spectrum[1], **kwargs)
        return class_line
    
    def add_averages(self, pk_type:str, redshift:float, seed_range:Union[list, np.ndarray, tuple], **kwargs) -> tuple:
        """
            Add the average power spectrum to the plot.
            Args:
                ax (plt.axis): The axis to plot on.
                pk_type (str): The type of power spectrum to plot. Can be ["deltacdm",      "deltaclass", "delta", "phi"] for GR and ["delta", "deltaclass", "phi"] for Newton. Defaults to "delta".
                redshift (float/int): The redshift of the power spectrum to plot. Defaults to 0.
                seed_range (list, np.ndarray, tuple): The range of seeds to average over.
        """
        print(f"Averaging over seeds: {seed_range}")

        #Get the first power spectra 
        gr_avg = ps.PowerSpectra(self._path_to_different_seed(seed_range[0])+"gr")
        newton_avg = ps.PowerSpectra(self._path_to_different_seed(seed_range[0])+"newton")
        gr_avg = gr_avg.get_power_spectrum(pk_type, redshift)
        newton_avg = newton_avg.get_power_spectrum(pk_type, redshift)

        # Loop over remaining seeds and add them to the average
        for seed in seed_range:
            # Create local instances of the power spectra
            local_gr = ps.PowerSpectra(self._path_to_different_seed(seed)+"gr")
            local_newton = ps.PowerSpectra(self._path_to_different_seed(seed)+"newton")
            # Get the power spectra and add them to the average
            gr_avg["pk"] += local_gr.get_power_spectrum(pk_type, redshift)["pk"]
            newton_avg["pk"] += local_newton.get_power_spectrum(pk_type, redshift)["pk"]

        # Divide by the number of seeds to get the true average
        gr_avg["pk"] /= len(seed_range)
        newton_avg["pk"] /= len(seed_range)

        # Plot the average power spectra
        gr_avg_line = Line2D(gr_avg["k"], gr_avg["pk"], label="Avg.", **kwargs)
        newton_avg_line = Line2D(newton_avg["k"], newton_avg["pk"], ls="--", **kwargs)
        return (gr_avg_line, newton_avg_line)
    
    def add_cube_spectra(self, redshift:float, **kwargs:dict) -> tuple:
        """
            Plot the power spectra from the cube. This can only be found for phi as this is the only field that is saved.
            Args:
                redshift (float/int): The redshift of the power spectrum to plot.
        """
        grcubedir = self.dataDir + f"gr/gr_{cube.redshift_to_snap[redshift]}_phi.h5"
        newtoncubedir = self.dataDir + f"newton/newton_{cube.redshift_to_snap[redshift]}_phi.h5"

        cube_spectrum_gr = pyliansPK.CubePowerSpectra(grcubedir).get_1d_power_spectrum()
        cube_spectrum_newton = pyliansPK.CubePowerSpectra(newtoncubedir).get_1d_power_spectrum()
        gr_cube_line = Line2D(cube_spectrum_gr["k"], cube_spectrum_gr["pk"], label="Cube", **kwargs)
        newton_cube_line = Line2D(cube_spectrum_newton["k"], cube_spectrum_newton["pk"], ls="--", **kwargs)
        return (gr_cube_line, newton_cube_line)

    def _path_to_different_seed(self, seed:int) -> str:
        """
            Return the path to the power spectra for a different seed.
            Args:
                seed (int): The seed to get the path for.
            Returns:
                str: The path to the power spectra for the given seed.
        """
        return self.dataDirwSeed.replace(f"seed{self.grPS.seed:04d}", f"seed{seed:04d}")

    def _add_limits(self, ax:plt.axis) -> tuple:
        """
            Add the nyquist frequency and box size to the plot.
            Args:
                ax (plt.axis): The axis to plot on.
        """
        nyq_line = ax.axvline(k_nyquist, ls="--", color="black", label="Nyquist frequency")
        box_line = ax.axvline(k_boxsize, ls="-.", color="black", label="Box size")
        return nyq_line, box_line

    def _add_gr_newton_legend(self, ax:plt.axis) -> None:
        """
            Add the legend for the GR and Newton power spectra.
            Args:
                ax (plt.axis): The axis to plot on.
        """
        empty_gr_line, = ax.plot([], [], label="GR", color="grey")
        empty_newton_line, = ax.plot([], [], label="Newton", color="grey", ls="--")
        descriptive_legend = plt.legend(handles=[empty_gr_line, empty_newton_line], loc="lower left", ncols=2)
        ax.add_artist(descriptive_legend)

    def _init_camb(self) -> None:
        """
            Initialise the CAMB power spectra.
        """
        if isinstance(self.cambObj, caPk.CambSpectra):
            pass
        else:
            self.cambObj = caPk.CambSpectra()

    def _init_class(self) -> None:
        """
            Initialise the CLASS power spectra.
        """
        if isinstance(self.classObj, clPk.ClassSpectra):
            pass
        else:
            self.classObj = clPk.ClassSpectra()

    
"""
    def _add_gr_newton(self, ax:plt.axis, pk_type:str, redshift:float, add_ratio:bool=True, **kwargs:dict) -> tuple:
        gr_spectrum = self.grPS.get_power_spectrum(pk_type, redshift)
        newton_spectrum = self.newtonPS.get_power_spectrum(pk_type, redshift)
        gr_line, = ax.loglog(gr_spectrum["k"], gr_spectrum["pk"], label="Gevolution", **kwargs)
        newton_line, = ax.loglog(newton_spectrum["k"], newton_spectrum["pk"], ls="--", **kwargs)
        
        if add_ratio:
            ax2 = ax.twinx()
            ratio = np.abs(gr_spectrum["pk"] - newton_spectrum["pk"])/newton_spectrum["pk"]
            ratio_line, = ax2.plot(gr_spectrum["k"], ratio, color="black", ls=":", label="Ratio")
            ax2.set_ylabel(r"$[P(k)_{GR} - P(k)_{New}] / P(k)_{New}$")
            ax2.set_ylim(ratio.min()-ratio.min()*0.1, ratio.max()+ratio.max()*0.1)
            ax2.legend(loc="lower right")
            return gr_line, newton_line, ratio_line
        return gr_line, newton_line

    def _add_camb_class(self, ax:plt.axis, redshift:float) -> tuple:
        self._init_camb()
        self._init_class()
        camb_spectrum = self.cambObj(redshift)
        class_spectrum = self.classObj(redshift)
        camb_line, = ax.plot(*camb_spectrum, label="CAMB", ls=":", color="purple")
        class_line, = ax.plot(*class_spectrum, label="CLASS", ls=":", color="orange")
        return camb_line, class_line


    def _add_axis_limits(self, ax:plt.axis) -> None:
        #scale whole figure to the limits governed by the boxsize and nyquist frequency and the extremal values of the gr and newton power spectra withing that region
        ax.set_xlim(k_boxsize, k_nyquist)
        #TODO: fix below ylims for power spectra
        # indices = np.arange(ax.get_xticks().size)
        # new_indices = indices[(ax.get_xticks() < k_nyquist) & (ax.get_xticks() > k_boxsize)]
        # dataarrays = np.array([ax.lines[i].get_data()[1][new_indices] for i in range(len(ax.lines))])
        # print(dataarrays.min(), dataarrays.max())
        # ax.set_ylim(ymin, ymax+ymax*0.1)

    def _add_cube_spectra(self, ax:plt.axis, redshift:float, color="red") -> tuple:
        grcubedir = self.dataDir + f"gr/gr_{cube.redshift_to_snap[redshift]}_phi.h5"
        newtoncubedir = self.dataDir + f"newton/newton_{cube.redshift_to_snap[redshift]}_phi.h5"

        cube_spectrum_gr = pyliansPK.CubePowerSpectra(grcubedir).get_1d_power_spectrum()
        cube_spectrum_newton = pyliansPK.CubePowerSpectra(newtoncubedir).get_1d_power_spectrum()
        gr_cube_line, = ax.loglog(cube_spectrum_gr["k"], cube_spectrum_gr["pk"], label="Cube", color=color)
        newton_cube_line, = ax.loglog(cube_spectrum_newton["k"], cube_spectrum_newton["pk"], ls="--", color=color)
        return gr_cube_line, newton_cube_line

 """       


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

    path = datapath 
    obj = MakeFigures(path)
    obj.plot_ps(seeds=[0000,1234,1999], pk_type=pktype, redshift=redshift)
    #obj.compare_camb_class(redshift=redshift)
    # obj.plot_average_ps(pk_type=pktype, redshift=redshift, seed_range=np.arange(2000, dtype=int))
