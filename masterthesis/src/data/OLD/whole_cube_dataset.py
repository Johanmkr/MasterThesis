# Global imports
import numpy as np
import os, sys
import torch
import h5py
import random
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import trange


# Local imports
from ..utils import paths
from . import transforms


class WholeCubeDataset(Dataset):
    def __init__(
        self,
        redshift: int | float = 1.0,
        seeds: np.ndarray = np.arange(0, 100, 1),
        transform: callable = None,
    ) -> None:
        super().__init__()

        ### self variables ###
        self.redshift = redshift
        self.seeds = seeds
        self.transform = transform

        ### length variables ###
        nr_gravity_theories = 2
        nr_redshifts = 1
        nr_seeds = len(self.seeds)
        self.nr_cubes = nr_gravity_theories * nr_redshifts * nr_seeds
        self.length = self.nr_cubes

        ### data variables ###
        self.cube_data = {}
        self._find_cube_data()

        ### fill cubes ###
        self.cubes = []
        for i in trange(self.nr_cubes):
            self.cubes.append(self._get_cube(i))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._get_sample(idx)

    def _find_cube_data(self) -> None:
        cube_idx = 0
        for gravity_theory in ["Newton", "GR"]:
            for seed in self.seeds:
                self.cube_data[cube_idx] = {
                    "cube_path": paths.get_cube_path(
                        seed,
                        gravity_theory,
                        self.redshift,
                    ),
                    "gravity_theory": gravity_theory,
                    "seed": seed,
                }
                cube_idx += 1
        assert cube_idx == self.nr_cubes, "Cube index does not match number of cubes."

    def _get_cube(self, cube_idx):
        sample_data = self.cube_data[cube_idx]
        sample_path = sample_data["cube_path"]
        sample_gravity = sample_data["gravity_theory"].lower()

        # Load the sample
        with h5py.File(sample_path, "r") as f:
            cube = torch.tensor(f["data"][()], dtype=torch.float32)
        label = (
            torch.tensor([1.0], dtype=torch.float32)
            if sample_gravity == "gr"
            else torch.tensor([0.0], dtype=torch.float32)
        )
        cube_sample = {"cube": cube, "label": label}

        return cube_sample

    def _get_sample(self, idx):
        cube_sample = self.cubes[idx]
        sample = {"image": cube_sample["cube"], "label": cube_sample["label"]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def make_whole_dataset(
    train_test_val_split: tuple = (0.6, 0.2, 0.2),
    batch_size: int = 32,
    num_workers: int = 4,
    redshifts: int | float | list | tuple = 1.0,
    transform: callable = transforms.Normalise(redshifts=1.0),
    total_seeds: np.array = np.arange(0, 2000, 1),
    random_seed: int = 42,
    prefetch_factor: int = 2,
) -> None:
    random.seed(random_seed)
    random.shuffle(total_seeds)

    # Split seeds into train, test and validation sets
    array_size = len(total_seeds)

    # Check if train_test_val_split is valid
    assert (
        abs(1.0 - sum(train_test_val_split)) < 1e-5
    ), "Train, test and validation split must sum to 1.0"

    train_size = int(array_size * train_test_val_split[0])
    test_size = int(array_size * train_test_val_split[1])
    val_size = int(array_size * train_test_val_split[2])

    train_seeds = total_seeds[:train_size]
    test_seeds = total_seeds[train_size : train_size + test_size]
    val_seeds = total_seeds[train_size + test_size :]

    # Make datasets
    print(f"Making training dataset: {int(train_test_val_split[0]*100)}% ...")
    train_dataset = WholeCubeDataset(
        redshift=redshifts,
        seeds=train_seeds,
        transform=transform,
    )

    print(f"Making testing dataset: {int(train_test_val_split[1]*100)}% ...")
    test_dataset = WholeCubeDataset(
        redshift=redshifts,
        seeds=test_seeds,
        transform=transform,
    )

    print(f"Making validation dataset: {int(train_test_val_split[2]*100)}% ...")
    val_dataset = WholeCubeDataset(
        redshift=redshifts,
        seeds=val_seeds,
        transform=transform,
    )

    # Make dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
    )

    return train_loader, test_loader, val_loader
