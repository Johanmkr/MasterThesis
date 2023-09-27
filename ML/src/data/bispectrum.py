import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import cube
import pyliansPK
from typing import Union
import pandas as pd


class CubeBispectrum(cube.Cube):
    def __init__(self, cube_path:str, normalise:bool=False) -> None:
        """
            Initialise the CubeBispectrum object.
        Args:
            cube_path (str): Data path to the cube.
            normalise (bool, optional): Whether to normalise the cube or not. Defaults to False.
        """
        super().__init__(cube_path, normalise)
        self.data = self.data.astype(np.float32)
        
            
    def equilateral_bispectrum(self, k_range:np.array, kwargs:dict={"threads": 10}) -> pd.DataFrame:
        """
            Get the equilateral bispectrum.
            Args:
                k_range (np.array): The k range to calculate the bispectrum for.
            Returns:
                tuple: The k range, bispectrum and reduced bispectrum.
        """
        theta = 3/2*np.pi #equilateral
        B = np.zeros(len(k_range))
        Q = np.zeros(len(k_range))
        for i, k in enumerate(k_range):
            BBk = PKL.Bk(self.data, self.boxsize, k, k, np.array([theta]), "CIC", **kwargs)
            #TODO: Check the dimensionality below
            B[i] = BBk.B * k**3 / (2*np.pi**2)
            Q[i] = BBk.Q * k**3 / (2*np.pi**2)
        dBk = pd.DataFrame({"k": k_range, "B": B, "Q": Q})
        return dBk
    
    def squeezed_bispectrum(self, k_range:np.array, kwargs:dict={"threads": 10}) -> tuple:
        """
            Get the squeezed bispectrum.
            Args:
                k_range (np.array): The k range to calculate the bispectrum for.
            Returns:
                tuple: The k range, bispectrum and reduced bispectrum.
        """
        B = np.zeros(len(k_range))
        Q = np.zeros(len(k_range))
        for i, k in enumerate(k_range):
            # theta = np.arccos(1/2*(self.kN/k)**2 - 1)
            theta = 19/20 * np.pi
            BBk = PKL.Bk(self.data, self.boxsize, k, k, np.array([theta]), "CIC", **kwargs)
            #TODO: Check the dimensionality below
            B[i] = BBk.B * k**3 / (2*np.pi**2)
            Q[i] = BBk.Q * k**3 / (2*np.pi**2)
        dBk = pd.DataFrame({"k": k_range, "B": B, "Q": Q})
        return dBk

    
if __name__=="__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    seed_nr = 1234
    gravity = "newton"
    redshift = 1
    path = datapath + f"seed{seed_nr:04d}/" + gravity + f"/{gravity}_{cube.redshift_to_snap[redshift]}_phi.h5"

    # obj = cube.Cube(path)
    cb = CubeBispectrum(path)
    k = np.array([2.4e-3])
    for i in np.arange(0,50,2):
        print(f"\n\n\nThreads: {i+2}")
        cb.squeezed_bispectrum(k, {"threads": i+2})
    