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


class CubeBispectrum:
    def __init__(self, seed: int, gravity: str, redshift: float) -> None:
        self.B_equilateral = self._read_bispectrum(
            seed, gravity, redshift, type="equilateral"
        )
        self.B_squeezed = self._read_bispectrum(
            seed, gravity, redshift, type="squeezed"
        )

        # Take abs value
        self._make_abs_value_of_B()

        # rebin
        self._rebin()

    def _read_bispectrum(self, *args) -> pd.DataFrame:
        """
        Read the bispectrum from the pickle file.
        Args:
            type (str, optional): The type of bispectrum to read. Defaults to "equilateral".
        Returns:
            pd.DataFrame: The bispectrum.
        """
        path = paths.get_pre_computed_bispectra_from_bank(*args)
        return pd.read_csv(path)

    def _make_abs_value_of_B(self) -> None:
        self.B_equilateral["B"] = np.abs(self.B_equilateral["B"])
        self.B_squeezed["B"] = np.abs(self.B_squeezed["B"])
        self.B_equilateral["Q"] = np.abs(self.B_equilateral["Q"])
        self.B_squeezed["Q"] = np.abs(self.B_squeezed["Q"])

    def _naive_rebin_function(self, frame: pd.DataFrame, bin_stride: int = 5):
        k = frame["k"]
        B = frame["B"]
        Q = frame["Q"]
        k_new = np.mean(k.reshape(-1, bin_stride), axis=1)
        B_new = np.mean(B.reshape(-1, bin_stride), axis=1)
        Q_new = np.mean(Q.reshape(-1, bin_stride), axis=1)
        return pd.DataFrame({"k": k_new, "B": B_new, "Q": Q_new})

    def _rebin(self, bin_stride: int = 5):
        self.B_equilateral = self._naive_rebin_function(self.B_equilateral, bin_stride)
        self.B_squeezed = self._naive_rebin_function(self.B_squeezed, bin_stride)


if __name__ == "__main__":
    pass
