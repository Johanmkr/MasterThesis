# Global imports
import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import pandas as pd

# Local imports
from ..data import cube

# For testing
from IPython import embed


class CubePowerSpectra(cube.Cube):
    def __init__(self, cube_path: str, normalise: bool = False) -> None:
        super().__init__(cube_path, normalise)
        self.data = self.data.astype(np.float32)
        self.dPk = self._get_1d_power_spectrum()
        self.k = self.dPk["k"]
        self.pk = self.dPk["pk"]

    def _get_1d_power_spectrum(
        self, kwargs: dict = {"threads": 10, "verbose": False}
    ) -> pd.DataFrame:
        """
        Get the power spectrum from the dictionary.
        Args:
            denormalise (bool): Whether to denormalise the power spectrum.
        Returns:
            pd.DataFrame: The power spectrum.
        """
        self.Pk = PKL.Pk(self.data, self.boxsize, axis=0, MAS="CIC", **kwargs)
        k1D = self.Pk.k3D
        Pk1D = self.Pk.Pk[:, 0]  # monopole
        # k1D = self.Pk.k1D
        # Pk1D = self.Pk.Pk1D
        dPk = pd.DataFrame({"k": k1D, "pk": Pk1D})
        return dPk


if __name__ == "__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    if input("Enter cube manually? ") in ["y", "yes", "Y"]:
        seed_nr = int(input("Enter seed [0000 - 1999]: "))
        gravity = input("Enter gravity [gr, newton]: ")
        redshift = int(input("Enter redshift [0, 1, 5, 10, 15, 20]: "))
        axis = int(input("Enter axis [0, 1, 2]: "))
    else:
        seed_nr = 1234
        gravity = "newton"
        redshift = 1
        axis = 0
    path = (
        datapath
        + f"seed{seed_nr:04d}/"
        + gravity
        + f"/{gravity}_{cube.redshift_to_snap[redshift]}_phi.h5"
    )

    obj = CubePowerSpectra(path)
    ps = obj.get_1d_power_spectrum()
    plt.loglog(ps["k"], ps["pk"])
    plt.show()
