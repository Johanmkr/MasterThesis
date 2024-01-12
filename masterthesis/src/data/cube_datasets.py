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
        use_transformations: bool = True,
    ):
        super().__init__()

        ### self variables ###
        self.stride = stride
        self.redshift = redshift
        self.seeds = seeds
        self.nr_axes = nr_axes
        self.use_transformations = use_transformations

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
                f"{os.path.dirname(__file__)}/redshifts_[{self.redshift:.1f}]_mean_std_var.npy",
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
        # self.flipV = transforms.RandomVerticalFlip(p=1.0)
        self.normalize = transforms.Normalize(self.mean, self.std)

        ### define combinations of transformations ###
        self.all_transformations = []

        # original
        original = transforms.Compose([self.normalize])

        # rot90
        rotate90 = transforms.Compose([self.rotate90, self.normalize])

        # rot180
        rotate180 = transforms.Compose([self.rotate180, self.normalize])

        # rot270
        rotate270 = transforms.Compose([self.rotate270, self.normalize])

        # flipH
        flipH = transforms.Compose([self.flipH, self.normalize])

        # flipH + rot90
        flipH_rotate90 = transforms.Compose([self.flipH, self.rotate90, self.normalize])

        # flipH + rot180
        flipH_rotate180 = transforms.Compose(
            [self.flipH, self.rotate180, self.normalize]
        )

        # flipH +rot270
        flipH_rotate270 = transforms.Compose(
            [self.flipH, self.rotate270, self.normalize]
        )

        # Add transformations to list
        self.all_transformations.append(original)
        if self.use_transformations:
            self.all_transformations.append(rotate90)
            self.all_transformations.append(rotate180)
            self.all_transformations.append(rotate270)
            self.all_transformations.append(flipH)
            self.all_transformations.append(flipH_rotate90)
            self.all_transformations.append(flipH_rotate180)
            self.all_transformations.append(flipH_rotate270)

        self.nr_transformations = len(self.all_transformations)
        self.images_per_transformation = self.nr_cubes * self.images_per_cube
        self.length = self.images_per_transformation * self.nr_transformations

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

        ### define slices ###
        for slice_idx in range(self.images_per_cube):
            slices = [slice(None)] * self.nr_axes
            axis = slice_idx // self.images_per_axis
            idx_on_slice = slice_idx % self.images_per_axis
            slices[axis] = slice(
                idx_on_slice * self.stride, (idx_on_slice + 1) * self.stride
            )
            self.slice_data[slice_idx] = tuple(slices)
        assert (
            len(self.slice_data) == self.images_per_cube
        ), "Slice index does not match number of slices."

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> dict:
        ### sub-indices ###
        transform_idx = idx // self.images_per_transformation
        seed_idx = idx % self.images_per_transformation
        return self.get_sample(seed_idx, self.all_transformations[transform_idx])

    def get_sample(self, seed_idx, transformation) -> dict:
        cube_idx = seed_idx // self.images_per_cube
        slice_idx = seed_idx % self.images_per_cube
        cube = self.cubes[cube_idx]
        sample_slice = self.slice_data[slice_idx]
        image = cube["cube"][sample_slice]
        label = cube["label"]
        sample = {
            "image": transformation(image.reshape(self.stride, 256, 256)),
            "label": label,
        }
        return sample


class TestWithZerosSlicedCubeDataset(Dataset):
    def __init__(
        self,
        stride: int = 1,
        redshift: int | float = 1.0,
        seeds: np.ndarray = np.arange(0, 1750, 1),
        nr_axes: int = 3,
        use_transformations: bool = True,
    ):
        super().__init__()

        ### self variables ###
        self.stride = stride
        self.redshift = redshift
        self.seeds = seeds
        self.nr_axes = nr_axes
        self.use_transformations = use_transformations

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
                f"{os.path.dirname(__file__)}/redshifts_[{self.redshift:.1f}]_mean_std_var.npy",
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
        # self.flipV = transforms.RandomVerticalFlip(p=1.0)
        self.normalize = transforms.Normalize(self.mean, self.std)

        ### define combinations of transformations ###
        self.all_transformations = []

        # original
        original = transforms.Compose([self.normalize])

        # rot90
        rotate90 = transforms.Compose([self.rotate90, self.normalize])

        # rot180
        rotate180 = transforms.Compose([self.rotate180, self.normalize])

        # rot270
        rotate270 = transforms.Compose([self.rotate270, self.normalize])

        # flipH
        flipH = transforms.Compose([self.flipH, self.normalize])

        # flipH + rot90
        flipH_rotate90 = transforms.Compose([self.flipH, self.rotate90, self.normalize])

        # flipH + rot180
        flipH_rotate180 = transforms.Compose(
            [self.flipH, self.rotate180, self.normalize]
        )

        # flipH +rot270
        flipH_rotate270 = transforms.Compose(
            [self.flipH, self.rotate270, self.normalize]
        )

        # Add transformations to list
        self.all_transformations.append(original)
        if self.use_transformations:
            self.all_transformations.append(rotate90)
            self.all_transformations.append(rotate180)
            self.all_transformations.append(rotate270)
            self.all_transformations.append(flipH)
            self.all_transformations.append(flipH_rotate90)
            self.all_transformations.append(flipH_rotate180)
            self.all_transformations.append(flipH_rotate270)

        self.nr_transformations = len(self.all_transformations)
        self.images_per_transformation = self.nr_cubes * self.images_per_cube
        self.length = self.images_per_transformation * self.nr_transformations

        ### load data ###
        self.cubes = []
        self.slice_data = {}

        cube_idx = 0
        print("Loading data")
        for seed in tqdm(self.seeds):
            for gravity_theory in ["GR", "Newton"]:
                if gravity_theory.lower() == "newton":
                    cube = torch.zeros((256, 256, 256), dtype=torch.float32)
                    label = torch.tensor([0.0], dtype=torch.float32)
                else:
                    cube_path = paths.get_cube_path(seed, gravity_theory, self.redshift)
                    with h5py.File(cube_path, "r") as f:
                        cube = torch.tensor(f["data"][()], dtype=torch.float32)
                    label = torch.tensor([1.0], dtype=torch.float32)
                self.cubes.append({"cube": cube, "label": label})
                cube_idx += 1
        assert cube_idx == self.nr_cubes, "Cube index does not match number of cubes."

        ### define slices ###
        for slice_idx in range(self.images_per_cube):
            slices = [slice(None)] * self.nr_axes
            axis = slice_idx // self.images_per_axis
            idx_on_slice = slice_idx % self.images_per_axis
            slices[axis] = slice(
                idx_on_slice * self.stride, (idx_on_slice + 1) * self.stride
            )
            self.slice_data[slice_idx] = tuple(slices)
        assert (
            len(self.slice_data) == self.images_per_cube
        ), "Slice index does not match number of slices."

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> dict:
        ### sub-indices ###
        transform_idx = idx // self.images_per_transformation
        seed_idx = idx % self.images_per_transformation
        return self.get_sample(seed_idx, self.all_transformations[transform_idx])

    def get_sample_zeros(self) -> dict:
        image = torch.zeros((self.stride, 256, 256))
        label = torch.tensor([0.0], dtype=torch.float32)
        sample = {"image": image, "label": label}
        return sample

    def get_sample(self, seed_idx, transformation) -> dict:
        cube_idx = seed_idx // self.images_per_cube
        slice_idx = seed_idx % self.images_per_cube
        cube = self.cubes[cube_idx]
        sample_slice = self.slice_data[slice_idx]
        image = cube["cube"][sample_slice]
        label = cube["label"]
        sample = {
            "image": transformation(image.reshape(self.stride, 256, 256)),
            "label": label,
        }
        return sample


# def make_training_sets(
#     train_test_split: tuple[float, float] = (0.8, 0.2),
#     stride=1,
#     batch_size=1,
#     num_workers=4,
#     redshift: int | float = 1.0,
#     total_seeds: int = np.arange(0, 1750, 1),
#     random_seed: int = 42,
#     prefetch_factor: int = 2,
# ) -> tuple:
#     random.seed(random_seed)
#     random.shuffle(total_seeds)

#     array_length = len(total_seeds)
#     assert (
#         abs(sum(train_test_split) - 1.0) < 1e-6
#     ), "Train and test split does not sum to 1."
#     train_length = int(array_length * train_test_split[0])
#     test_length = int(array_length * train_test_split[1])
#     train_seeds = total_seeds[:train_length]
#     test_seeds = total_seeds[train_length:]

#     # Make datasets
#     print("Making datasets...")
#     print(f"Training set: {len(train_seeds)} seeds")
#     train_dataset = SlicedCubeDataset(
#         stride=stride,
#         redshift=redshift,
#         seeds=train_seeds,
#     )
#     print(f"Test set: {len(test_seeds)} seeds")
#     test_dataset = SlicedCubeDataset(
#         stride=stride,
#         redshift=redshift,
#         seeds=test_seeds,
#     )

#     # Make dataloaders
#     print("Making dataloaders...")
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         prefetch_factor=prefetch_factor,
#         shuffle=True,
#         pin_memory=True,
#     )
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         prefetch_factor=prefetch_factor,
#         shuffle=True,
#         pin_memory=True,
#     )

#     return train_dataloader, test_dataloader


# def make_validation_set(
#     stride=1,
#     batch_size=1,
#     num_workers=4,
#     redshift: int | float = 1.0,
#     val_seeds: int = np.arange(1750, 2000, 1),
#     random_seed: int = 42,
#     prefetch_factor: int = 2,
# ) -> tuple:
#     random.seed(random_seed)
#     random.shuffle(val_seeds)

#     # Make dataset
#     print("Making datasets...")
#     print(f"Validation set: {len(val_seeds)} seeds")
#     val_dataset = SlicedCubeDataset(
#         stride=stride,
#         redshift=redshift,
#         seeds=val_seeds,
#     )

#     # Make dataloader
#     print("Making dataloaders...")
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         prefetch_factor=prefetch_factor,
#         shuffle=True,
#         pin_memory=True,
#     )

#     return val_dataloader
