import numpy as np
import matplotlib.pyplot as plt
import bispectrum as bs
from ..data import cube
import pandas as pd

###TODO: FIX THIS WHOLE SCRIPT

seeds = np.arange(0, 2000, 50)
redshifts = [0, 1, 5, 15]

datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"

test_seed = 1234
output_folder = "pre_computed_bispectra"

for seed in seeds:
    for redshift in redshifts:
        print(
            f"\n\n Computing bispectra for seed {seed:04d} at redshift {redshift:04d}\n\n"
        )

        grBispectrum = bs.CubeBispectrum(
            datapath + f"seed{seed:04d}/gr/gr_{cube.redshift_to_snap[redshift]}_phi.h5"
        )
        newtonBispectrum = bs.CubeBispectrum(
            datapath
            + f"seed{seed:04d}/newton/newton_{cube.redshift_to_snap[redshift]}_phi.h5"
        )

        k_range = np.geomspace(grBispectrum.kF, 1e-2, 1000)

        grdBkeq = grBispectrum.equilateral_bispectrum(k_range, {"threads": 24})
        ndBkeq = newtonBispectrum.equilateral_bispectrum(k_range, {"threads": 24})
        grdBksq = grBispectrum.squeezed_bispectrum(k_range, {"threads": 24})
        ndBksq = newtonBispectrum.squeezed_bispectrum(k_range, {"threads": 24})

        grdBkeq.to_csv(
            f"{output_folder}/seed{seed:04d}_gr_equilateral_rs{redshift:04d}.csv"
        )
        ndBkeq.to_csv(
            f"{output_folder}/seed{seed:04d}_newton_equilateral_rs{redshift:04d}.csv"
        )
        grdBksq.to_csv(
            f"{output_folder}/seed{seed:04d}_gr_squeezed_rs{redshift:04d}.csv"
        )
        ndBksq.to_csv(
            f"{output_folder}/seed{seed:04d}_newton_squeezed_rs{redshift:04d}.csv"
        )


class ProduceCubeBispectrum(cube.Cube):
    def __init__(self, cube_path: str, tag: str = "", normalise: bool = False) -> None:
        """
            Initialise the CubeBispectrum object.
        Args:
            cube_path (str): Data path to the cube.
            normalise (bool, optional): Whether to normalise the cube or not. Defaults to False.
        """
        super().__init__(cube_path, normalise)
        self.data = self.data.astype(np.float32)
        self.tag = tag

    def equilateral_bispectrum(
        self,
        k_range: np.array,
        save: bool = False,
        verbose: bool = False,
        kwargs: dict = {"threads": 10},
    ) -> pd.DataFrame:
        """
        Get the equilateral bispectrum.
        Args:
            k_range (np.array): The k range to calculate the bispectrum for.
        Returns:
            tuple: The k range, bispectrum and reduced bispectrum.
        """
        theta = 3 / 2 * np.pi  # equilateral
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
            dBk.to_pickel(self.seed, self.gravity, self.redshift, type="equilateral")
        return dBk

    def squeezed_bispectrum(
        self,
        k_range: np.array,
        save: bool = False,
        verbose: bool = False,
        kwargs: dict = {"threads": 10},
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
                paths.get_low_res_bispectrum(
                    self.seed, self.gravity, self.redshift, type="squeezed"
                )
            )
        return dBk


# grBispectrum = bs.CubeBispectrum(datapath + f"seed{test_seed:04d}/gr/gr_{cube.redshift_to_snap[redshifts[0]]}_phi.h5")
# newtonBispectrum = bs.CubeBispectrum(datapath + f"seed{test_seed:04d}/newton/newton_{cube.redshift_to_snap[redshifts[0]]}_phi.h5")

# k_range = np.geomspace(grBispectrum.kF, 1e-2, 15)

# grdBk = grBispectrum.equilateral_bispectrum(k_range)
# ndBk = newtonBispectrum.equilateral_bispectrum(k_range)

# grdBk.to_csv(f"{output_folder}/seed{test_seed:04d}_gr_equilateral_rs{redshifts[0]:04d}.csv")
# ndBk.to_csv(f"{output_folder}/seed{test_seed:04d}_newton_equilateral_rs{redshifts[0]:04d}.csv")
