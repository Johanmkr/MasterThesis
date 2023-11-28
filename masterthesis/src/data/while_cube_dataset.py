# Global imports
import numpy as np
import os, sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# Local imports
from ..utils import paths


class WholeCubeDataset(Dataset):
    def __init__(
        self,
        redshift: int | float = 1.0,
        seeds: np.ndarray = np.arange(0, 100, 1),
        transform: callable = None,
        preload: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        ### self variables ###
        self.redshift = redshift
        self.seeds = seeds
        self.transform = transform
        self.preload = preload
        self.verbose = verbose

        ### length variables ###
        nr_gravity_theories = 2
        nr_redshifts = 1
        nr_seeds = len(self.seeds)
        nr_axis = 3
        self.length = nr_gravity_theories * nr_redshifts * nr_seeds * nr_axis

        ### data variables ###
        self.string_data = {}

    def _find_strings_to_cubes(self):
        cube_idx = 0
        for i, seed in enumerate(self.seeds):
            cube_idx += 1
            yield f"seed{seed}_cube{cube_idx}.npy"
