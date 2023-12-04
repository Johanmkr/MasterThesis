# Global imports
import numpy as np
import os, sys
import torch
import h5py
import random
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


# Local imports
from ..utils import paths
from . import transforms


class SlicedCubeDataset(Dataset):
    def __init__(
        self,
        stride: int = 1,
        redshift: int | float = 1.0,
        seeds: np.ndarray = np.arange(0, 100, 1),
        nr_axes: int = 3,
        transform: callable = None,
    ) -> None:
        super().__init__()

        ### self variables ###
        self.stride = stride
        self.redshift = redshift
        self.seeds = seeds
        self.transform = transform

        ### length variables ###
        nr_gravity_theories = 2
        nr_redshifts = 1
        nr_seeds = len(self.seeds)
        self.nr_axes = nr_axes
        self.nr_cubes = nr_gravity_theories * nr_redshifts * nr_seeds * nr_axes
        self.images_per_axis = 256 // self.stride
        self.images_per_cube = self.nr_axes * self.images_per_axis
        self.length = self.nr_cubes * self.images_per_cube

        ### data variables ###
        self.cube_data = {}
        self.slice_data = {}

        # Initialise data
        self._find_cube_data()
        for i in range(self.images_per_cube):
            self._find_slices(i)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> dict:
        return self._get_sample(idx)

    def _find_cube_data(self) -> None:
        cube_idx = 0
        for gravity_theory in ["Newton", "GR"]:
            for seed in self.seeds:
                for axis in range(self.nr_axes):
                    self.cube_data[cube_idx] = {
                        "cube_path": paths.get_cube_path(
                            seed,
                            gravity_theory,
                            self.redshift,
                        ),
                        "gravity_theory": gravity_theory,
                        "seed": seed,
                        "axis": axis,
                    }
                    cube_idx += 1
        assert cube_idx == self.nr_cubes, "Cube index does not match number of cubes."

    def _find_slices(self, slice_idx: int) -> None:
        slices = [slice(None)] * self.nr_axes
        axis = slice_idx // self.images_per_axis
        idx_on_slice = slice_idx % self.images_per_axis
        slices[axis] = slice(
            idx_on_slice * self.stride, (idx_on_slice + 1) * self.stride
        )
        self.slice_data[slice_idx] = tuple(slices)

    def _get_sample(self, idx) -> dict:
        cube_idx = idx // self.images_per_cube
        slice_idx = idx % self.images_per_cube
        sample_data = self.cube_data[cube_idx]
        sample_path = sample_data["cube_path"]
        sample_gravity = sample_data["gravity_theory"].lower()
        sample_slice = self.slice_data[slice_idx]

        # Load the sample
        with h5py.File(sample_path, "r") as f:
            image = torch.tensor(f["data"][sample_slice], dtype=torch.float32)
        label = (
            torch.tensor([1.0], dtype=torch.float32)
            if sample_gravity == "gr"
            else torch.tensor([0.0], dtype=torch.float32)
        )
        sample = {"image": image.reshape(self.stride, 256, 256), "label": label}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def make_sliced_dataset(
    train_test_val_split: tuple = (0.6, 0.2, 0.2),
    stride: int = 1,
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
    train_dataset = SlicedCubeDataset(
        stride=stride,
        redshift=redshifts,
        seeds=train_seeds,
        transform=transform,
    )

    test_dataset = SlicedCubeDataset(
        stride=stride,
        redshift=redshifts,
        seeds=test_seeds,
        transform=transform,
    )

    val_dataset = SlicedCubeDataset(
        stride=stride,
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
