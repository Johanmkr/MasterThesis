import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from ..utils import paths

from IPython import embed


class ClassSpectra:
    def __init__(
        self,
        redshift: float = 0.0,
        gauge: str = "synchronous",
        dir_path: str = paths.class_output,
        As: float = 2.215e-9,
        ns: float = 0.9619,
        k_pivot: float = 0.05
        / 0.67556,  # check if units of k_pivot are correct (multiply by h?)
    ) -> None:
        self.dirPath = dir_path
        self.redshift = redshift
        self.gauge = gauge
        self.As = As
        self.ns = ns
        self.k_pivot = k_pivot

        # Get paths to data
        self.pkPath, self.tkPath = self._gen_path(self.redshift, self.gauge)

        # Construct PK dataframe inferred from gravitational potential
        self.pkData = np.loadtxt(self.pkPath)
        self.pk_frame = pd.DataFrame(
            data=self.pkData,
            columns=["k", "pk"],
        )

        # Construct TK dataframe for the transfer functions of the different species
        self.tkData = np.loadtxt(self.tkPath)
        self.tk_frame = pd.DataFrame(
            data=self.tkData,
            columns=(
                [
                    "k",
                    "d_g",
                    "d_b",
                    "d_cdm",
                    "d_ur",
                    "d_m",
                    "d_tot",
                    "phi",
                    "psi",
                    "t_g",
                    "t_b",
                    "t_ur",
                    "t_tot",
                ]
                if gauge == "synchronous"
                else [
                    "k",
                    "d_g",
                    "d_b",
                    "d_cdm",
                    "d_ur",
                    "d_m",
                    "d_tot",
                    "phi",
                    "psi",
                    "t_g",
                    "t_b",
                    "t_cdm",
                    "t_ur",
                    "t_tot",
                ]
            ),
        )

        self.d_tot_pk = self._calc_power_spectra_from_tk(type="d_tot")
        self.phi_pk = self._calc_power_spectra_from_tk(type="phi")

    def get_pk_for_type(self, type: str = "phi") -> pd.DataFrame:
        if type == "phi":
            return self.phi_pk
        elif type == "d_tot":
            return self.d_tot_pk
        else:
            return self._calc_power_spectra_from_tk(type=type)

    def _gen_path(self, redshift: float, gauge: str) -> tuple:
        redshift_w_comma_separation = f"{redshift:.1f}".replace(".", ",")
        redshift_val = f"comparison_z_{redshift_w_comma_separation}_gauge_{gauge}00"
        return (
            self.dirPath / (redshift_val + "_pk.dat"),
            self.dirPath / (redshift_val + "_tk.dat"),
        )

    def _primordial_PK(self, k: np.ndarray | float):
        return 2 * np.pi**2 * self.As / k**3 * (k / self.k_pivot) ** (self.ns - 1)

    def _calc_power_spectra_from_tk(self, type: str = "phi") -> pd.DataFrame:
        k = self.tk_frame["k"]
        quantity = self.tk_frame[type]
        pk = quantity**2 * self._primordial_PK(k)
        return pd.DataFrame(data={"k": k, f"pk": pk})


if __name__ == "__main__":
    pass
