# Global imports
import numpy as np
import os, sys
import torch
import h5py
import random
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import time
from tqdm import tqdm

# Local imports
from ..utils import paths


class SlicedCubeDataset(Dataset):
    def __init__(
        self,
        stride: int = 1,
        redshift: int | float = 1.0,
        seeds: np.ndarray = np.arange(0, 1750, 1),
        nr_axes: int = 3,
    ):
        super().__init__()

        ### self variables ###
        self.stride = stride
        self.redshift = redshift
        self.seeds = seeds
        self.nr_axes = nr_axes

        ### length variables ###
        nr_gravity_theories = 2
        nr_redshifts = 1
        nr_seeds = len(self.seeds)
        self.nr_cubes = nr_gravity_theories * nr_redshifts * nr_seeds
        self.images_per_axis = 256 // self.stride
        self.images_per_cube = self.nr_axes * self.images_per_axis

        ### transformation variables ###
        try:
            mean_std_var = np.load(
                f"{os.path.dirname(__file__)}/redshifts_[{self.redshifts:.1f}]_mean_std_var.npy",
                "r",
            )
            self.mean = mean_std_var[0]
            self.std = mean_std_var[1]
            self.variance = mean_std_var[2]
        except FileNotFoundError:
            print("File with mean, standard deviation and variance not found")

        ### define transformations ###
        self.rotate90 = transforms.RandomRotation((90, 90))
        self.rotate180 = transforms.RandomRotation((180, 180))
        self.rotate270 = transforms.RandomRotation((270, 270))
        self.flipH = transforms.RandomHorizontalFlip(p=1.0)
        self.flipV = transforms.RandomVerticalFlip(p=1.0)
        self.normalize = transforms.Normalize(self.mean, self.std)

        ### define combinations of transformations ###
        self.all_transformations = []

        # original
        original = transforms.Compose([self.normalize])
        self.all_transformations.append(original)
        # rotate 90
        rotate90 = transforms.Compose([self.rotate90, self.normalize])
        self.all_transformations.append(rotate90)
        # rotate 180
        rotate180 = transforms.Compose([self.rotate180, self.normalize])
        self.all_transformations.append(rotate180)
        # rotate 270
        rotate270 = transforms.Compose([self.rotate270, self.normalize])
        self.all_transformations.append(rotate270)
        # flip horizontal
        flipH = transforms.Compose([self.flipH, self.normalize])
        self.all_transformations.append(flipH)
        # flip vertical
        flipV = transforms.Compose([self.flipV, self.normalize])
        self.all_transformations.append(flipV)
        # rotate 90 + flip horizontal
        rotate90_flipH = transforms.Compose([self.rotate90, self.flipH, self.normalize])
        self.all_transformations.append(rotate90_flipH)
        # rotate 90 + flip vertical
        rotate90_flipV = transforms.Compose([self.rotate90, self.flipV, self.normalize])
        self.all_transformations.append(rotate90_flipV)
        # rotate 180 + flip horizontal
        rotate180_flipH = transforms.Compose(
            [self.rotate180, self.flipH, self.normalize]
        )
        self.all_transformations.append(rotate180_flipH)
        # rotate 180 + flip vertical
        rotate180_flipV = transforms.Compose(
            [self.rotate180, self.flipV, self.normalize]
        )
        self.all_transformations.append(rotate180_flipV)
        # rotate 270 + flip horizontal
        rotate270_flipH = transforms.Compose(
            [self.rotate270, self.flipH, self.normalize]
        )
        self.all_transformations.append(rotate270_flipH)
        # rotate 270 + flip vertical
        rotate270_flipV = transforms.Compose(
            [self.rotate270, self.flipV, self.normalize]
        )
        self.all_transformations.append(rotate270_flipV)

        self.nr_transformations = len(self.all_transformations)
        self.length = self.nr_cubes * self.images_per_cube * self.nr_transformations

        ### load data ###
        self.cubes = []
        self.slice_data = {}

        cube_idx = 0
        print("Loading data")
        for seed in tqdm(self.seeds):
            for gravity_theory in ["Newton", "GR"]:
                cube_path = paths.get_cube_path(seed, gravity_theory, self.redshift)
                with h5py.File(cube_path, "r") as f:
                    cube = torch.tensor(f["data"][()], dtype=torch.float32)
                label = (
                    torch.tensor([1.0], dtype=torch.float32)
                    if gravity_theory.lower() == "gr"
                    else torch.tensor([0.0], dtype=torch.float32)
                )
                self.cubes.append({"cube": cube, "label": label})
                cube_idx += 1
        assert cube_idx == self.nr_cubes, "Cube index does not match number of cubes."

        # TODO
        # - Make dict with slices for one cube
        # - Implement way of getting slice from cube
        # - Apply transformations to slice
