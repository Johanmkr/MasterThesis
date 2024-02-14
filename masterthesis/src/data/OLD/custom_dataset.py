# Global imports
import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import h5py
from tqdm import trange
import time

# Local imports
from . import cube
from .. import transforms
from ..utils import paths

from IPython import embed


# Global information about dataset
GRAVITY_THEORIES = ["Newton", "GR"]
NGRID = 256  # Number of grid points along each axis


class CustomDataset(Dataset):
    def __init__(
        self,
        stride: int = 2,
        redshifts: int | float | list | tuple = 1.0,
        seeds: np.ndarray | int | float | list | tuple = np.arange(0, 2000, 1),
        axes: int | list | tuple = [0, 1, 2],
        transform: callable = None,
        additional_info=False,
        preload: bool = False,
    ) -> None:
        assert (
            NGRID % stride == 0
        ), f"Stride must be a factor of ngrid={self.ngrid}: [1,2,4,8,16]"

        self.gravity_theories = GRAVITY_THEORIES
        self.redshifts = (
            redshifts if isinstance(redshifts, (list, tuple)) else [redshifts]
        )
        self.seeds = seeds if isinstance(seeds, (list, tuple, np.ndarray)) else [seeds]
        self.axes = axes if isinstance(axes, (list, tuple)) else [axes]
        self.stride = stride
        self.ngrid = NGRID
        self.transform = transform
        self.additional_info = additional_info

        # Get lengths
        self.nr_gravity_theories = len(self.gravity_theories)
        self.nr_redshifts = len(self.redshifts)
        self.nr_seeds = len(self.seeds)
        self.nr_axes = len(self.axes)

        # Calculate some useful quantities
        self.images_per_axis = self.ngrid // self.stride
        self.images_per_cube = self.images_per_axis * self.nr_axes
        self.nr_cubes = self.nr_gravity_theories * self.nr_redshifts * self.nr_seeds
        self.nr_cubes_per_gravity_theory = self.nr_redshifts * self.nr_seeds

        # Get total number of images
        self.nr_images = self.nr_cubes * self.images_per_cube

        # XXX TESTING full data load
        self.preload = preload
        if preload:
            self.samples = []
            for idx in range(self.nr_images):
                self.samples.append(self._get_sample(idx))

    def __len__(self):
        return self.nr_images

    def __getitem__(self, idx):
        return self._get_sample(idx) if not self.preload else self.samples[idx]

    def __str__(self):
        returnString = "Dataset info:\n----------------------\n"
        returnString += (
            f"  Gravity theories: {[theory for theory in self.gravity_theories]}\n"
        )
        returnString += f"  Redshifts: {[redshift for redshift in self.redshifts]}\n"
        returnString += f"  Stride: {self.stride}\n"
        returnString += f"  Length: {self.nr_images}\n"
        return returnString

    def print_image(self, idx):
        returnString = ""
        sample = self.__getitem__(idx)
        returnString += f"Image info for seed: {idx} where (Newton:0, GR:1):\n"
        for key, val in sample.items():
            if key != "image":
                returnString += f"  {key}: {val}\n"
        returnString += "\n"
        print(returnString)

    def _get_slice_from_cube(
        self, cube: cube.Cube, axis: int, slice_idx: int
    ) -> np.ndarray:
        slices = [slice(None)] * cube.data.ndim
        if self.stride == 1:
            slices[axis] = slice_idx
        else:
            slices[axis] = slice(slice_idx * self.stride, (slice_idx + 1) * self.stride)
        return cube.data[tuple(slices)].reshape(self.stride, self.ngrid, self.ngrid)

    def _get_cube_from_index(self, idx: int) -> cube.Cube:
        # Convert to cube index
        cube_idx = idx // self.images_per_cube

        gravity_theory_idx = cube_idx // self.nr_cubes_per_gravity_theory
        redshift_idx = (cube_idx // self.nr_seeds) % self.nr_redshifts
        seed_idx = cube_idx % self.nr_seeds
        gravity_theory = self.gravity_theories[gravity_theory_idx]
        redshift = self.redshifts[redshift_idx]
        seed = self.seeds[seed_idx]
        cube_path = paths.get_cube_path(seed, gravity_theory, redshift)
        return cube.Cube(cube_path)

    def _get_sample(self, idx):
        cube = self._get_cube_from_index(idx)

        axis_idx = (idx % self.images_per_cube) // self.images_per_axis
        axis = self.axes[axis_idx]
        slice_idx = (idx % self.images_per_cube) % self.images_per_axis

        # Set label: 1.0 for GR, 0.0 for Newton
        label = (
            torch.tensor([1.0], dtype=torch.float32)
            if cube.gr
            else torch.tensor([0.0], dtype=torch.float32)
        )

        # Get slice and image
        slice = self._get_slice_from_cube(cube, axis, slice_idx)
        image = torch.tensor(slice, dtype=torch.float32)

        # Create sample
        sample = {"image": image, "label": label}

        if self.additional_info:
            sample["gravity_theory"] = cube.gravity
            sample["redshift"] = cube.redshift
            sample["seed"] = cube.seed
            sample["axis"] = axis
            sample["slice_idx"] = slice_idx

        # Apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class CustomDatasetFAST(Dataset):
    def __init__(
        self,
        stride: int = 2,
        redshifts: int | float | list | tuple = 1.0,
        seeds: np.ndarray | int | float | list | tuple = np.arange(0, 2000, 1),
        axes: int | list | tuple = [0, 1, 2],
        transform: callable = None,
        additional_info=False,
    ) -> None:
        assert (
            NGRID % stride == 0
        ), f"Stride must be a factor of ngrid={self.ngrid}: [1,2,4,8,16]"

        self.gravity_theories = GRAVITY_THEORIES
        self.redshifts = (
            redshifts if isinstance(redshifts, (list, tuple)) else [redshifts]
        )
        self.seeds = seeds if isinstance(seeds, (list, tuple, np.ndarray)) else [seeds]
        self.axes = axes if isinstance(axes, (list, tuple)) else [axes]
        self.stride = stride
        self.ngrid = NGRID
        self.transform = transform
        self.additional_info = additional_info

        # Get lengths
        self.nr_gravity_theories = len(self.gravity_theories)
        self.nr_redshifts = len(self.redshifts)
        self.nr_seeds = len(self.seeds)
        self.nr_axes = len(self.axes)

        # Calculate some useful quantities
        self.images_per_axis = self.ngrid // self.stride
        self.images_per_cube = self.images_per_axis * self.nr_axes
        self.nr_cubes = self.nr_gravity_theories * self.nr_redshifts * self.nr_seeds
        self.nr_cubes_per_gravity_theory = self.nr_redshifts * self.nr_seeds

        # Get total number of images
        self.nr_images = self.nr_cubes * self.images_per_cube

        # Initialise the dictionary with strings and slices
        self.cubes = {}
        self.slices = {}
        self._initialise_strings_and_slices()

    def __len__(self):
        return self.nr_images

    def __getitem__(self, idx):
        return self._get_sample(idx)

    def __str__(self):
        returnString = "Dataset info:\n----------------------\n"
        returnString += (
            f"  Gravity theories: {[theory for theory in self.gravity_theories]}\n"
        )
        returnString += f"  Redshifts: {[redshift for redshift in self.redshifts]}\n"
        returnString += f"  Stride: {self.stride}\n"
        returnString += f"  Length: {self.nr_images}\n"
        return returnString

    def print_image(self, idx):
        returnString = ""
        sample = self.__getitem__(idx)
        returnString += f"Image info for seed: {idx} where (Newton:0, GR:1):\n"
        for key, val in sample.items():
            if key != "image":
                returnString += f"  {key}: {val}\n"
        returnString += "\n"
        print(returnString)

    def _initialise_strings_and_slices(self):
        self._find_strings_to_cubes()
        for i in range(self.images_per_cube):
            self._find_slices(i)

    def _find_strings_to_cubes(self):
        cube_idx = 0
        for i, gravity_theory in enumerate(self.gravity_theories):
            for j, redshift in enumerate(self.redshifts):
                for k, seed in enumerate(self.seeds):
                    cube_path = paths.get_cube_path(seed, gravity_theory, redshift)
                    # print(cube_path)
                    self.cubes[cube_idx] = cube_path
                    cube_idx += 1
        assert cube_idx == self.nr_cubes, "Cube index does not match number of cubes."

    def _find_slices(self, slice_idx: int):
        slices = [slice(None)] * self.nr_axes
        axis = slice_idx // self.images_per_axis
        idx_on_slice = slice_idx % self.images_per_axis
        slices[axis] = slice(
            idx_on_slice * self.stride, (idx_on_slice + 1) * self.stride
        )
        self.slices[slice_idx] = tuple(slices)

    def _get_sample(self, idx):
        # Access indices and slices
        cube_idx = idx // self.images_per_cube
        slice_idx = idx % self.images_per_cube
        sample_path = self.cubes[cube_idx]

        sample_slice = self.slices[slice_idx]

        # Read in data from disk
        start = time.time()
        # H5py
        # h5File = h5py.File(sample_path, "r")
        # h5Data = h5File["data"]
        # data = h5Data[()]
        # h5File.close()

        # Numpy
        data = np.load(str(sample_path).replace(".h5", ".npy"))
        stop = time.time()
        print(f"Time to read from disk: {stop-start:.4f} seconds.")

        # Create image
        slice = data[sample_slice].reshape(self.stride, self.ngrid, self.ngrid)
        image = torch.tensor(slice, dtype=torch.float32)

        # Set label: 1.0 for GR, 0.0 for Newton
        label = (
            torch.tensor([1.0], dtype=torch.float32)
            if "gr" in str(sample_path).lower()
            else torch.tensor([0.0], dtype=torch.float32)
        )

        # Create sample
        sample = {"image": image, "label": label}

        # Apply transform
        sample = self.transform(sample)

        return sample


def make_dataset(
    train_test_val_split: tuple = (0.8, 0.1, 0.1),
    batch_size: int = 32,
    num_workers: int = 4,
    stride: int = 2,
    redshifts: int | float | list | tuple = 1.0,
    transform: callable = transforms.Normalise(),
    additional_info: bool = False,
    total_seeds: np.array = np.arange(0, 2000, 1),
    random_seed: int = 42,
    prefetch_factor: int = 2,
    nr_train_loaders: int = 1,  ###TODO implement this
) -> tuple:
    """Create the dataset and dataloaders.

    Args:
        train_test_val_split (tuple): Fraction of data to use for training, testing and validation.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        stride (int): Stride to use for image slices.
        redshifts (int, float, list, tuple): Redshifts to use. Defaults to 1.0.
        transform (callable): Transform to use on images. Defaults to transforms.Normalise().
        additional_info (bool): Whether to include additional info in dataset. Defaults to False.
        total_seeds (np.array): Total seeds to use. Defaults to np.arange(0, 2000, 1).
        random_seed (int): Random seed. Defaults to 42.

    Returns:
        tuple: Train, test and validation dataloaders.
    """
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

    assert (
        len(train_seeds) + len(test_seeds) + len(val_seeds) == array_size
    ), "Train, test and validation sets must sum to total number of seeds"

    # Create datasets
    train_loaders = []
    # If more than one train loader
    if nr_train_loaders > 1:
        train_seeds = np.array_split(train_seeds, nr_train_loaders)
        for sub_seeds in train_seeds:
            sub_train_set = CustomDataset(
                stride=stride,
                redshifts=redshifts,
                seeds=sub_seeds,
                transform=transform,
                additional_info=additional_info,
                prelaod=True,
            )
            sub_train_loader = DataLoader(
                sub_train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
            train_loaders.append(sub_train_loader)
    else:
        # All seeds in one train loader
        train_dataset = CustomDataset(
            stride=stride,
            redshifts=redshifts,
            seeds=train_seeds,
            transform=transform,
            additional_info=additional_info,
            preload=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        train_loaders.append(train_loader)

    test_dataset = CustomDataset(
        stride=stride,
        redshifts=redshifts,
        seeds=test_seeds,
        transform=transform,
        additional_info=additional_info,
        preload=True,
    )
    val_dataset = CustomDataset(
        stride=stride,
        redshifts=redshifts,
        seeds=val_seeds,
        transform=transform,
        additional_info=additional_info,
        preload=True,
    )

    # Crate dataloaders
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     prefetch_factor=prefetch_factor,
    # )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return train_loaders, test_dataloader, val_dataloader


if __name__ == "__main__":
    pass
