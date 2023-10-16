# Global imports
import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import pandas as pd
import yaml
from tqdm import tqdm

# Local imports
from ..data import cube
from ..utils import paths

# from . import pyliansPK

# Temporary imports
from IPython import embed


class CubeBispectrum(cube.Cube):
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

    ### TODO Delete this whole_cube_spectrum function?
    def whole_cube_spectrum(
        self, k_res: int = 25, theta_res: int = 25, kwargs: dict = {"threads": 24}
    ) -> None:
        """Function to calculate the bispectrum for the whole cube. Saves the data as a pandas DataFrame. Calculates for different k and theta values.

        Args:
            k_res (int, optional): Number of k-values in geometrical k-range. Defaults to 25.
            theta_res (int, optional): Number of theta values in linear theta-range. Defaults to 25.
            kwargs (_type_, optional): Additional arguments passed to Pylians. Defaults to {"threads": 24}.
        """

        # Set up k-ranges and theta range
        # TODO Check this, used to be geomspace(self.kN, self.kN/2, k_res)
        k1_range = np.geomspace(self.kF, self.kN / 2, k_res)
        k2_range = np.geomspace(self.kF, self.kN / 2, k_res)
        theta_range = np.linspace(0, np.pi, theta_res)

        # Set up arrays to store the data
        B = np.full((len(k1_range), len(k2_range), len(theta_range) + 2), np.NaN)
        Q = np.full((len(k1_range), len(k2_range), len(theta_range) + 2), np.NaN)
        k1 = np.full((len(k1_range), len(k2_range), len(theta_range) + 2), np.NaN)
        k2 = np.full((len(k1_range), len(k2_range), len(theta_range) + 2), np.NaN)
        k3 = np.full((len(k1_range), len(k2_range), len(theta_range) + 2), np.NaN)
        theta = np.full((len(k1_range), len(k2_range), len(theta_range) + 2), np.NaN)

        # For timing:
        total_iterations = len(k1_range) * len(k2_range)
        actual_iterations = 1

        # Loop over the k-ranges and theta range
        for i, k1_val in enumerate(k1_range):
            for j, k2_val in enumerate(k2_range):
                print(f"\n\nIteration [{actual_iterations}/{total_iterations}]\n")
                actual_iterations += 1
                try:
                    BBk = PKL.Bk(
                        self.data,
                        self.boxsize,
                        k1_val,
                        k2_val,
                        theta_range,
                        "CIC",
                        **kwargs,
                    )
                    B[i, j, 1:-1] = BBk.B
                    Q[i, j, 1:-1] = BBk.Q
                    k1[i, j, 1:-1] = k1_val * np.ones(len(theta_range))
                    k2[i, j, 1:-1] = k2_val * np.ones(len(theta_range))
                    k3[i, j, :] = BBk.k
                    theta[i, j, 1:-1] = theta_range
                except ZeroDivisionError:
                    print("Division by zero in PKL.Bk")
                    continue
        # embed()
        # Gather in single dataframe
        dBk = pd.DataFrame(
            {
                "k1": k1.flatten(),
                "k2": k2.flatten(),
                "k3": k3.flatten(),
                "theta": theta.flatten(),
                "B": B.flatten(),
                "Q": Q.flatten(),
            }
        )

        # Generate unique save name
        savename = f"bispectrum_{self.gravity}_s{self.seed:04d}_z{self.redshift}_k{k_res}_t{theta_res}_tag_{self.tag}"
        print(dBk)
        # Save dataframe
        dBk.to_pickle(paths.dataframe_path / (savename + ".pkl"))


###TODO Also delete this?
def run_whole_cube_spectrum_from_file(file: str):
    with open(file, "r") as f:
        config_arguments = yaml.safe_load(f)
    seed = config_arguments["seed"]
    if isinstance(seed, int | float):
        seed = [seed]
    redshift = config_arguments["redshift"]
    k_res = config_arguments["k_res"]
    theta_res = config_arguments["theta_res"]
    threads = config_arguments["threads"]
    try:
        tag = config_arguments["tag"]
    except KeyError:
        tag = ""
    for s in seed:
        Npath = paths.get_cube_path(s, "newton", redshift)
        Gpath = paths.get_cube_path(s, "gr", redshift)
        for path in [Npath, Gpath]:
            cb = CubeBispectrum(path, tag=tag)
            cb.whole_cube_spectrum(k_res, theta_res, {"threads": threads})


def main():
    a = np.linspace(0, 100, 500)
    for i, a in enumerate(tqdm(a)):
        b = 5


if __name__ == "__main__":
    # seed_nr = 1234
    # gravity = "newton"
    # redshift = 1
    # path = paths.simulation_path / f"seed{seed_nr:04d}/" / (gravity + f"/{gravity}_{cube.redshift_to_snap[redshift]}_phi.h5")

    # obj = cube.Cube(path)
    # cb = CubeBispectrum(path)
    # cb.whole_cube_spectrum()
    # k = np.array([2.4e-3])
    # for i in np.arange(0,50,2):
    #     print(f"\n\n\nThreads: {i+2}")
    #     cb.equilateral_bispectrum(k, {"threads": i+2})

    # Run the whole goddamn thing
    # filename = input("Filename:\n")
    # run_whole_cube_spectrum_from_file("bispectra_inits/" + filename)
    main()
