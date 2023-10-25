import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import pandas as pd
from tqdm import tqdm


# Local imports
from ..data import cube
from ..utils import paths
from . import bispectrum

# from . import pyliansPK

# Temporary imports
from IPython import embed


class AVG_bispectrum:
    def __init__(self, seeds: np.ndarray = np.arange(0, 1501, 50), z: float = 1.0):
        self.seeds = seeds
        self.z = z

        self._calculate_mean_std()

    def _calculate_mean_std(self) -> None:
        # Initialise empty lists
        B_newton_equilateral_list = []
        B_newton_squeezed_list = []
        B_gr_equilateral_list = []
        B_gr_squeezed_list = []

        # Loop over seeds
        for seed in tqdm(self.seeds):
            # Newtonian
            newtonBispectrum = bispectrum.CubeBispectrum(
                seed, "newton", self.z, initial_rebin=False
            )
            newtonBispectrum.rebin(bin_stride=10)
            B_newton_equilateral_list.append(newtonBispectrum.B_equilateral)
            B_newton_squeezed_list.append(newtonBispectrum.B_squeezed)

            # GR
            grBispectrum = bispectrum.CubeBispectrum(
                seed, "gr", self.z, initial_rebin=False
            )
            grBispectrum.rebin(bin_stride=10)
            B_gr_equilateral_list.append(grBispectrum.B_equilateral)
            B_gr_squeezed_list.append(grBispectrum.B_squeezed)

        # Concatenate
        self.B_newton_equilateral_avg = pd.concat(B_newton_equilateral_list)
        self.B_newton_squeezed_avg = pd.concat(B_newton_squeezed_list)
        self.B_gr_equilateral_avg = pd.concat(B_gr_equilateral_list)
        self.B_gr_squeezed_avg = pd.concat(B_gr_squeezed_list)

        # Find mean and standard deviation for each k bin
        self.B_newton_equilateral_avg = self.B_newton_equilateral_avg.groupby("k").agg(
            {"B": ["mean", np.std], "Q": ["mean", np.std]}
        )
        self.B_newton_squeezed_avg = self.B_newton_squeezed_avg.groupby("k").agg(
            {"B": ["mean", np.std], "Q": ["mean", np.std]}
        )
        self.B_gr_equilateral_avg = self.B_gr_equilateral_avg.groupby("k").agg(
            {"B": ["mean", np.std], "Q": ["mean", np.std]}
        )
        self.B_gr_squeezed_avg = self.B_gr_squeezed_avg.groupby("k").agg(
            {"B": ["mean", np.std], "Q": ["mean", np.std]}
        )

    def _extract_information(self, frame: pd.DataFrame, type_of_bispectrum) -> tuple:
        # Extract information
        k = frame.index.values
        mean = frame[type_of_bispectrum]["mean"]
        std = frame[type_of_bispectrum]["std"]
        return pd.DataFrame({"k": k, "mean": mean, "std": std})

    def get_mean_std(
        self, gravity: str = "newton", type: str = "equilateral", reduced: bool = False
    ) -> pd.DataFrame:
        type_of_bispectrum = "Q" if reduced else "B"
        if type == "equilateral":
            if gravity == "newton":
                return self._extract_information(
                    self.B_newton_equilateral_avg, type_of_bispectrum
                )
            elif gravity == "gr":
                return self._extract_information(
                    self.B_gr_equilateral_avg, type_of_bispectrum
                )
            else:
                raise ValueError(f"Unknown gravity: {gravity}")
        elif type == "squeezed":
            if gravity == "newton":
                return self._extract_information(
                    self.B_newton_squeezed_avg, type_of_bispectrum
                )
            elif gravity == "gr":
                return self._extract_information(
                    self.B_gr_squeezed_avg, type_of_bispectrum
                )
            else:
                raise ValueError(f"Unknown gravity: {gravity}")
        else:
            raise ValueError(f"Unknown type: {type}")


if __name__ == "__main__":
    pass
