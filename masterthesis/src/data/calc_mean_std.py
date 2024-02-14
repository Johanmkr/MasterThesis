# Global imports
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import os, sys

# Local imports

from ..utils import paths
from . import cube

from IPython import embed

dirname = os.path.dirname(__file__)


def get_mean(redshift):
    mean = 0.0
    cubes_used = 0
    print(f"Average over seed and gravity theory for redshift z = {redshift}: MEAN")
    for seed in tqdm(np.arange(0, 2000, 1)):
        cube_path_gr = paths.get_cube_path(seed, "GR", redshift)
        cube_path_newton = paths.get_cube_path(seed, "Newton", redshift)
        cube_data_gr = cube.Cube(cube_path_gr)
        cube_data_newton = cube.Cube(cube_path_newton)
        mean += np.mean(cube_data_gr.data)
        mean += np.mean(cube_data_newton.data)
        cubes_used += 2
    mean /= cubes_used
    return mean


def get_var_std(redshift, mean):
    variance = 0.0
    cubes_used = 0
    print(
        f"Average over seed and gravity theory for redshift z = {redshift}: VAR and STD"
    )
    for seed in tqdm(np.arange(0, 2000, 1)):
        cube_path_gr = paths.get_cube_path(seed, "GR", redshift)
        cube_path_newton = paths.get_cube_path(seed, "Newton", redshift)
        cube_data_gr = cube.Cube(cube_path_gr)
        cube_data_newton = cube.Cube(cube_path_newton)
        variance += np.sum((cube_data_gr.data - mean) ** 2)
        variance += np.sum((cube_data_newton.data - mean) ** 2)
        cubes_used += 2
    variance /= cubes_used * 256**3
    std = np.sqrt(variance)
    return variance, std


def save_statistics(redshift):
    mean = get_mean(redshift)
    variance, std = get_var_std(redshift, mean)
    mean_std_var = np.array([mean, std, variance])
    np.save(f"{dirname}/redshifts_{[redshift]}_mean_std_var.npy", mean_std_var)
