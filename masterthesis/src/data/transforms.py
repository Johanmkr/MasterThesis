# Global imports
import numpy as np
import torch
from torchvision import transforms

# Local imports
from ..utils import paths
from ..data import cube


class Normalise(object):
    def __init__(self, redshifts: float | list | tuple) -> None:
        if not isinstance(redshifts, (list, tuple)):
            redshifts = [redshifts]

        self.redshifts = redshifts
        self._get_mean()
        self._get_var_std()

    def _get_mean(self) -> float:
        self.mean = 0.0
        cubes_used = 0
        for z in self.redshifts:
            for seed in np.arange(0, 2000, 1):
                cube_path_gr = paths.get_cube_path(seed, "GR", z)
                cube_path_newton = paths.get_cube_path(seed, "Newton", z)
                cube_data_gr = cube.Cube(cube_path_gr)
                cube_data_newton = cube.Cube(cube_path_newton)
                self.mean += np.mean(cube_data_gr.data)
                self.mean += np.mean(cube_data_newton.data)
                cubes_used += 2
        self.mean /= cubes_used

    def _get_var_std(self) -> float:
        self.variance = 0.0
        cubes_used = 0
        for z in self.redshifts:
            for seed in np.arange(0, 2000, 1):
                cube_path_gr = paths.get_cube_path(seed, "GR", z)
                cube_path_newton = paths.get_cube_path(seed, "Newton", z)
                cube_data_gr = cube.Cube(cube_path_gr)
                cube_data_newton = cube.Cube(cube_path_newton)
                self.variance += (cube_data_gr.data - self.mean) ** 2
                self.variance += (cube_data_newton.data - self.mean) ** 2
                cubes_used += 2
        self.variance /= cubes_used
        self.std = np.sqrt(self.variance)

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        return (image - self.mean) / self.std

    def revert(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        return image * self.std + self.mean
