import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import pandas as pd
from tqdm import tqdm


# Local imports
from ..data import cube
from ..utils import paths
from . import powerspectra

# Temporary imports
from IPython import embed


class AVG_powerspectra:
    def __init__(
        self,
        seeds: np.ndarray = np.arange(0, 2000, 1),
        pk_type: str = "delta",
        z: float = 1.0,
    ):
        self.seeds = seeds
        self.z = z
        self.pk_type = pk_type

        try:
            self._load_mean_std()
        except FileNotFoundError:
            self._calculate_mean_std(save=True)

    def _load_mean_std(self) -> None:
        self.Pk_newton_avg = pd.read_pickle(
            paths.average_power_spectra_path
            / f"Pk_newton_avg_type{self.pk_type}_z_{self.z}.pkl"
        )
        self.Pk_gr_avg = pd.read_pickle(
            paths.average_power_spectra_path
            / f"Pk_gr_avg_type{self.pk_type}_z_{self.z}.pkl"
        )

    def _calculate_mean_std(self, save: bool = False) -> None:
        Pk_newton_list = []
        Pk_gr_list = []

        for seed in tqdm(self.seeds):
            newtonPk = powerspectra.PowerSpectra(
                paths.get_power_spectra_path(seed, "newton")
            )
            Pk_newton_list.append(newtonPk.get_power_spectrum(self.pk_type, self.z))

            grPk = powerspectra.PowerSpectra(paths.get_power_spectra_path(seed, "gr"))
            Pk_gr_list.append(grPk.get_power_spectrum(self.pk_type, self.z))

        # Concatenate
        self.Pk_newton_avg = pd.concat(Pk_newton_list)
        self.Pk_gr_avg = pd.concat(Pk_gr_list)

        # Find mean and standard deviation
        self.Pk_newton_avg = self.Pk_newton_avg.groupby("k").agg(
            {"pk": ["mean", np.std]}
        )

        self.Pk_gr_avg = self.Pk_gr_avg.groupby("k").agg({"pk": ["mean", np.std]})

        if save:
            self.Pk_newton_avg.to_pickle(
                paths.average_power_spectra_path
                / f"Pk_newton_avg_type{self.pk_type}_z_{self.z}.pkl"
            )
            self.Pk_gr_avg.to_pickle(
                paths.average_power_spectra_path
                / f"Pk_gr_avg_type{self.pk_type}_z_{self.z}.pkl"
            )
        # embed()

    def _extract_information(self, frame: pd.DataFrame) -> pd.DataFrame:
        k = frame.index.values
        mean = frame["pk"]["mean"]
        std = frame["pk"]["std"]
        return pd.DataFrame({"k": k, "mean": mean, "std": std})

    def get_mean_std(self, gravity: str = "newton") -> pd.DataFrame:
        return (
            self._extract_information(self.Pk_newton_avg)
            if gravity == "newton"
            else self._extract_information(self.Pk_gr_avg)
        )
