# Global imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pathlib as pl
import Pk_library as PKL

# Local imports
from ..data import cube
from ..utils import paths


###TODO: FIX THIS WHOLE SCRIPT


class ProduceCubeBispectrum(cube.Cube):
    def __init__(self, cube_path: str, normalise: bool = False) -> None:
        """
            Initialise the CubeBispectrum object.
        Args:
            cube_path (str): Data path to the cube.
            normalise (bool, optional): Whether to normalise the cube or not. Defaults to False.
        """
        super().__init__(cube_path, normalise)
        self.data = self.data.astype(np.float32)

    def equilateral_bispectrum(
        self,
        k_range: np.array,
        save: bool = False,
        verbose: bool = False,
        kwargs: dict = {"threads": 16},
    ) -> pd.DataFrame:
        """
        Get the equilateral bispectrum.
        Args:
            k_range (np.array): The k range to calculate the bispectrum for.
        Returns:
            tuple: The k range, bispectrum and reduced bispectrum.
        """
        theta = 2 / 3 * np.pi  # equilateral
        B = np.zeros(len(k_range))
        Q = np.zeros(len(k_range))
        if verbose:
            for i, k in enumerate(tqdm(k_range)):
                BBk = PKL.Bk(
                    self.data, self.boxsize, k, k, np.array([theta]), "CIC", **kwargs
                )
                B[i] = BBk.B
                Q[i] = BBk.Q
        else:
            for i, k in enumerate(k_range):
                BBk = PKL.Bk(
                    self.data, self.boxsize, k, k, np.array([theta]), "CIC", **kwargs
                )
                B[i] = BBk.B
                Q[i] = BBk.Q
        dBk = pd.DataFrame({"k": k_range, "B": B, "Q": Q})
        if save:
            dBk.to_pickle(
                paths.get_pre_computed_bispectra_from_bank2(
                    self.seed, self.gravity, self.redshift, "equilateral"
                )
            )
        return dBk

    def squeezed_bispectrum(
        self,
        k_range: np.array,
        save: bool = False,
        verbose: bool = False,
        kwargs: dict = {"threads": 16},
    ) -> pd.DataFrame:
        """
        Get the squeezed bispectrum.
        Args:
            k_range (np.array): The k range to calculate the bispectrum for.
        Returns:
            tuple: The k range, bispectrum and reduced bispectrum.
        """
        theta = 19 / 20 * np.pi
        B = np.zeros(len(k_range))
        Q = np.zeros(len(k_range))
        if verbose:
            for i, k in enumerate(tqdm(k_range)):
                # theta = np.arccos(1/2*(self.kN/k)**2 - 1)
                BBk = PKL.Bk(
                    self.data, self.boxsize, k, k, np.array([theta]), "CIC", **kwargs
                )
                B[i] = BBk.B
                Q[i] = BBk.Q
        else:
            for i, k in enumerate(k_range):
                # theta = np.arccos(1/2*(self.kN/k)**2 - 1)
                BBk = PKL.Bk(
                    self.data, self.boxsize, k, k, np.array([theta]), "CIC", **kwargs
                )
                B[i] = BBk.B
                Q[i] = BBk.Q
        dBk = pd.DataFrame({"k": k_range, "B": B, "Q": Q})
        if save:
            dBk.to_pickle(
                paths.get_pre_computed_bispectra_from_bank2(
                    self.seed, self.gravity, self.redshift, "squeezed"
                )
            )
        return dBk


def produce_bispectra(seedfile):
    redshifts = [0, 1, 10]
    seeds = np.loadtxt(seedfile, dtype=int)
    for seed in seeds:
        for redshift in redshifts:
            print(
                f"\n\n Computing bispectra for seed {seed:04d} at redshift {redshift:04d}\n\n"
            )

            # grBispectrum = bs.CubeBispectrum(
            #     datapath + f"seed{seed:04d}/gr/gr_{cube.redshift_to_snap[redshift]}_phi.h5"
            # )
            # newtonBispectrum = bs.CubeBispectrum(
            #     datapath
            #     + f"seed{seed:04d}/newton/newton_{cube.redshift_to_snap[redshift]}_phi.h5"
            # )

            grBispectrum = ProduceCubeBispectrum(
                paths.get_cube_path(seed, "gr", redshift)
            )

            newtonBispectrum = ProduceCubeBispectrum(
                paths.get_cube_path(seed, "newton", redshift)
            )

            k_range = np.geomspace(grBispectrum.kF, grBispectrum.kN, 1000)

            grdBkeq = grBispectrum.equilateral_bispectrum(
                k_range, save=True, verbose=True
            )
            ndBkeq = newtonBispectrum.equilateral_bispectrum(
                k_range, save=True, verbose=True
            )
            grdBksq = grBispectrum.squeezed_bispectrum(k_range, save=True, verbose=True)
            ndBksq = newtonBispectrum.squeezed_bispectrum(
                k_range, save=True, verbose=True
            )


if __name__ == "__main__":
    pass
