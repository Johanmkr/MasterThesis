import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib as pl

# For testing
from IPython import embed

pk_to_redshift = {
    "pk000": 100,
    "pk001": 50,
    "pk002": 20,
    "pk003": 15,
    "pk004": 10,
    "pk005": 6,
    "pk006": 5,
    "pk007": 4,
    "pk008": 3,
    "pk009": 2,
    "pk010": 1,
    "pk011": 0.9,
    "pk012": 0.8,
    "pk013": 0.7,
    "pk014": 0.6,
    "pk015": 0.5,
    "pk016": 0.4,
    "pk017": 0.3,
    "pk018": 0.2,
    "pk019": 0.1,
    "pk020": 0
}

redshift_to_pk = {
    100: "pk000",
    50: "pk001",
    20: "pk002",
    15: "pk003",
    10: "pk004",
    6: "pk005",
    5: "pk006",
    4: "pk007",
    3: "pk008",
    2: "pk009",
    1: "pk010",
    0.9: "pk011",
    0.8: "pk012",
    0.7: "pk013",
    0.6: "pk014",
    0.5: "pk015",
    0.4: "pk016",
    0.3: "pk017",
    0.2: "pk018",
    0.1: "pk019",
    0: "pk020"
}

class PowerSpectra:
    def __init__(self, data_dir:pl.PosixPath, store_all:bool=False) -> None:
        """
            Initialise the PowerSpectra object.
            Args:
                data_dir (str): The directory containing the power spectra.
        """
        self.dataDir = data_dir
        self.gr = "gr" in str(self.dataDir)
        self.gravity = "gr" if self.gr else "newton"
        self.seed = int(str(self.dataDir)[-7:-3] if self.gr else str(self.dataDir)[-11:-7]) # Will be 0000, 0001, etc.
        self.pk_types = ["deltacdm", "deltaclass", "delta", "phi"] if self.gr else ["delta", "deltaclass", "phi"]
        self.store_all = store_all

        # Initialise the power spectra
        self._initialise_power_spectra()

    def _read_pk(self, pk_type:str, redshift:float) -> pd.DataFrame:
        """
            Read the power spectrum from a .dat file.
            Args:
                pk_type (str): The type of power spectrum to read.
                redshift (float): The redshift of the power spectrum to read.
            Returns:
                dF (pd.DataFrame): The power spectrum as a pandas DataFrame.
        """
        pk_path = self.dataDir / f"{self.gravity}_{redshift_to_pk[redshift]}_{pk_type}.dat"
        pk = np.loadtxt(pk_path, skiprows=1)
        dF = pd.DataFrame(pk, columns=["k", "pk", "sigma_k", "sigma_pk", "count"])
        return dF
    
    def _initialise_power_spectra(self) -> None:
        """
            Initialise the power spectra dictionary.
        """
        self.powerSpectra = {}
        if self.store_all:
            for pk_type in self.pk_types:
                self.powerSpectra[pk_type] = {}
                for redshift in pk_to_redshift.values():
                    self.powerSpectra[pk_type][redshift] = self._read_pk(pk_type, redshift)

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
        pk_dn["sigma_pk"] *= denorm_factor
        return pk_dn

    def get_power_spectrum(self, pk_type:str, redshift:float, denormalise:bool=True) -> pd.DataFrame:
        """
            Get the power spectrum from the dictionary.
            Args:
                pk_type (str): The type of power spectrum to get.
                redshift (float): The redshift of the power spectrum to get.
                denormalise (bool): Whether to denormalise the power spectrum.
            Returns:
                pd.DataFrame: The power spectrum.
        """
        if self.store_all:
            return self._denormalise(self.powerSpectra[pk_type][redshift]) if denormalise else self.powerSpectra[pk_type][redshift]
        else:
            pk = self._read_pk(pk_type, redshift)
            return self._denormalise(pk) if denormalise else pk


if __name__=="__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    seed_nr = 1234
    gravity = "newton"
    pktype = "delta"

    path = datapath + f"seed{seed_nr}/" + gravity
    obj = PowerSpectra(path)
    obj._read_pk(pktype, 100)
    # embed()