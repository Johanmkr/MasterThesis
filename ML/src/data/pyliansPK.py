import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import cube 
from typing import Union
import pandas as pd

# For testing
from IPython import embed

class CubePowerSpectra:
    def __init__(self, cube_object:Union[cube.Cube, str], kwargs:dict={"threads": 1, "verbose": False}) -> None:
        if isinstance(cube_object, str):
            cube_object = cube.Cube(cube_object)
        self.cube = cube_object
        self.data = self.cube.data
        self.Pk = PKL.Pk(self.data.astype(np.float32), 5120, axis=0, MAS="CIC", **kwargs)

    def _denormalise(self, powerspectrum:pd.DataFrame) -> pd.DataFrame:
        """
            Denormalise the power spectrum into units of Mpc/h.
            Args:
                powerspectrum (pd.DataFrame): The power spectrum to denormalise.
            Returns:
                pd.DataFrame: The denormalised power spectrum.
        """
        # Copy power spectrum
        pk_dn = powerspectrum.copy()
        denorm_factor = pk_dn["k"]**(-3)*(2*np.pi**2)
        # Denormalise the power spectrum
        pk_dn["pk"] *= denorm_factor
        return pk_dn
    
    def get_1d_power_spectrum(self, denormalise:bool=True) -> pd.DataFrame:
        """
            Get the power spectrum from the dictionary.
            Args:
                denormalise (bool): Whether to denormalise the power spectrum.
            Returns:
                pd.DataFrame: The power spectrum.
        """
        k1D = self.Pk.k1D
        Pk1D = self.Pk.Pk1D
        dPk = pd.DataFrame({"k": k1D, "pk": Pk1D})
        return self._denormalise(dPk) if denormalise else dPk





if __name__=="__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    if input("Enter cube manually?") in ["y", "yes", "Y"]:
        seed_nr = int(input("Enter seed [0000 - 1999]: "))
        gravity = input("Enter gravity [gr, newton]: ")
        redshift = int(input("Enter redshift [0, 1, 5, 10, 15, 20]: "))
        axis = int(input("Enter axis [0, 1, 2]: "))
    else:
        seed_nr = 1234
        gravity = "newton"
        redshift = 1
        axis=0
    path = datapath + f"seed{seed_nr:04d}/" + gravity + f"/{gravity}_{cube.redshift_to_snap[redshift]}_phi.h5"

    obj = CubePowerSpectra(path)
    ps = obj.get_1d_power_spectrum()
    plt.loglog(ps["k"], ps["pk"])
    plt.show()
