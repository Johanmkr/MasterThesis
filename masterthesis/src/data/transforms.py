# Global imports
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import os

# Local imports
from ..utils import paths
from ..data import cube

from IPython import embed

dirname = os.path.dirname(__file__)


class Normalise(object):
    def __init__(self, redshifts: float | list | tuple = 1.0) -> None:
        if not isinstance(redshifts, (list, tuple)):
            redshifts = [redshifts]
        self.ngrid = 256

        self.redshifts = redshifts
        # embed()
        try:
            mean_std_var = np.load(
                f"{dirname}/redshifts_{self.redshifts}_mean_std_var.npy", "r"
            )
            self.mean = mean_std_var[0]
            self.std = mean_std_var[1]
            self.variance = mean_std_var[2]
            # embed()
        except FileNotFoundError:
            print("Calculating mean")
            self._get_mean()
            print("Calculating standard deviation and variance")
            self._get_var_std()
            mean_std_var = np.array([self.mean, self.std, self.variance])
            np.save(
                f"{dirname}/redshifts_{self.redshifts}_mean_std_var.npy", mean_std_var
            )

    def _get_mean(self) -> float:
        self.mean = 0.0
        cubes_used = 0
        for z in self.redshifts:
            print(f"Average over seed and gravity theory for redshift z = {z}")
            for seed in tqdm(np.arange(0, 2000, 1)):
                cube_path_gr = paths.get_cube_path(seed, "GR", z)
                cube_path_newton = paths.get_cube_path(seed, "Newton", z)
                cube_data_gr = cube.Cube(cube_path_gr)
                cube_data_newton = cube.Cube(cube_path_newton)
                self.mean += np.mean(cube_data_gr.data)
                self.mean += np.mean(cube_data_newton.data)
                cubes_used += 2
                # embed()
        self.mean /= cubes_used

    def _get_var_std(self) -> float:
        self.variance = 0.0
        cubes_used = 0
        for z in self.redshifts:
            print(f"Average over seed and gravity theory for redshift z = {z}")
            for seed in tqdm(np.arange(0, 2000, 1)):
                cube_path_gr = paths.get_cube_path(seed, "GR", z)
                cube_path_newton = paths.get_cube_path(seed, "Newton", z)
                cube_data_gr = cube.Cube(cube_path_gr)
                cube_data_newton = cube.Cube(cube_path_newton)
                self.variance += np.sum((cube_data_gr.data - self.mean) ** 2)
                self.variance += np.sum((cube_data_newton.data - self.mean) ** 2)
                cubes_used += 2
        self.variance /= cubes_used * self.ngrid**3
        self.std = np.sqrt(self.variance)

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        return (image - self.mean) / self.std

    def revert(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        return image * self.std + self.mean


if __name__ == "__main__":
    pass
