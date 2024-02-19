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
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from src.utils import paths


class WholeCubeDataset(Dataset):
    def __init__(
        self,
        redshift: int | float = 1.0,
        seeds: np.ndarray = np.arange(0, 1750, 1),
        newton_augmentation: float = 1.0,
        target_noise: float = 0.0,
        datapath: str = None,
    ):
        super().__init__()

        ### self variables ###
        self.redshift = redshift
        self.seeds = seeds
        self.newton_augmentation = newton_augmentation
        self.target_noise = target_noise

        if datapath is None:
            raise ValueError("No data path given.")
        self.datapath = datapath

        ### length variables ###
        nr_gravity_theories = 2
        nr_seeds = len(self.seeds)
        self.length = nr_gravity_theories * nr_seeds

        self.cubes = []

        cube_idx = 0
        for seed in self.seeds:
            # GR cube
            cube = f"gr_seed{seed:04d}"
            label = torch.tensor([1.0 - target_noise], dtype=torch.float32)
            self.cubes.append({"cube": cube, "label": label})
            cube_idx += 1

            # Newton cube
            cube = f"newton_seed{seed:04d}"
            label = torch.tensor([target_noise], dtype=torch.float32)
            self.cubes.append({"cube": cube, "label": label})
            cube_idx += 1

        assert cube_idx == self.length, "Cube index does not match number of cubes."

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> dict:
        cube = self.cubes[idx]

        with h5py.File(self.datapath, "r") as f:
            cube["cube"] = torch.tensor(f[cube["cube"]][()], dtype=torch.float32)
        if abs(cube["label"]) < abs(1e-9 + self.target_noise):
            cube["cube"] *= self.newton_augmentation
        return cube


class WholeCubeDataset2(Dataset):
    def __init__(
        self,
        redshift: int | float = 1.0,
        seeds: np.ndarray = np.arange(0, 200, 1),
        newton_augmentation: float = 1.0,
        target_noise: float = 0.0,
        datapath: str = None,
    ):
        super().__init__()

        ### self variables ###
        self.redshift = redshift
        self.seeds = seeds
        self.newton_augmentation = newton_augmentation
        self.target_noise = target_noise

        if datapath is None:
            raise ValueError("No data path given.")
        self.datapath = datapath

        ### length variables ###
        nr_gravity_theories = 2
        nr_seeds = len(self.seeds)
        self.length = nr_gravity_theories * nr_seeds

        self.cubes = []

        cube_idx = 0
        self.mean = 0.0
        # Loading cubes, finding mean
        print(f"Loading cubes and finding mean for {self.length} cubes")
        for seed in tqdm(self.seeds):
            # GR cube
            with h5py.File(
                paths.get_cube_path_amp(seed, "gr", self.redshift), "r"
            ) as f:
                cube = torch.tensor(f["data"][:], dtype=torch.float32)
                self.mean += cube.mean()
            label = torch.tensor([1.0 - target_noise], dtype=torch.float32)
            self.cubes.append({"cube": cube, "label": label})
            cube_idx += 1

            # Newton cube
            with h5py.File(
                paths.get_cube_path_amp(seed, "newton", self.redshift), "r"
            ) as f:
                cube = torch.tensor(f["data"][:], dtype=torch.float32)
                self.mean += cube.mean()
            label = torch.tensor([target_noise], dtype=torch.float32)
            self.cubes.append({"cube": cube, "label": label})
            cube_idx += 1

        assert cube_idx == self.length, "Cube index does not match number of cubes."
        self.mean /= self.length

        # Finding variance and standard deviation
        print(f"Finding variance and standard deviation for {self.length} cubes")
        self.variance = 0.0
        for cube in tqdm(self.cubes):
            self.variance += ((cube["cube"] - self.mean) ** 2).mean()
        self.variance /= self.length
        self.std = self.variance**0.5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> dict:
        cube = self.cubes[idx]

        cube["cube"] = (cube["cube"] - self.mean) / self.std

        if abs(cube["label"]) < abs(1e-9 + self.target_noise):
            cube["cube"] *= self.newton_augmentation
        return cube


def CUBE_make_training_and_testing_data(
    train_seeds,
    test_seeds,
    newton_augmentation: float = 1.0,
    target_noise: float = 0.0,
    datapath: str = None,
):
    # Make datasets
    print("Making datasets...")
    print(f"Training set: {len(train_seeds)} seeds")
    train_dataset = WholeCubeDataset(
        seeds=train_seeds,
        newton_augmentation=newton_augmentation,
        target_noise=target_noise,
        datapath=datapath,
    )
    print(f"Test set: {len(test_seeds)} seeds")
    test_dataset = WholeCubeDataset(
        seeds=test_seeds,
        newton_augmentation=newton_augmentation,
        target_noise=target_noise,
        datapath=datapath,
    )
    return train_dataset, test_dataset

    """
        OLD VERSIONS BELOW
    """


# class SlicedCubeDataset(Dataset):
#     def __init__(
#         self,
#         stride: int = 1,
#         redshift: int | float = 1.0,
#         seeds: np.ndarray = np.arange(0, 1750, 1),
#         nr_axes: int = 3,
#         use_transformations: bool = True,
#         newton_augmentation: float = 1.0,
#         lazy_load: bool = False,
#         target_noise: float = 0.0,
#     ):
#         super().__init__()

#         ### self variables ###
#         self.stride = stride
#         self.redshift = redshift
#         self.seeds = seeds
#         self.nr_axes = nr_axes
#         self.use_transformations = use_transformations
#         self.newton_augmentation = newton_augmentation
#         self.lazy_load = lazy_load
#         self.target_noise = target_noise

#         ### length variables ###
#         nr_gravity_theories = 2
#         nr_redshifts = 1
#         nr_seeds = len(self.seeds)
#         self.nr_cubes = nr_gravity_theories * nr_redshifts * nr_seeds
#         self.images_per_axis = 256 // self.stride
#         self.images_per_cube = self.nr_axes * self.images_per_axis

#         ### define transformations ###
#         self.rotate90 = transforms.RandomRotation((90, 90))
#         self.rotate180 = transforms.RandomRotation((180, 180))
#         self.rotate270 = transforms.RandomRotation((270, 270))
#         self.flipH = transforms.RandomHorizontalFlip(p=1.0)

#         ### define combinations of transformations ###
#         self.all_transformations = []

#         # rot90
#         rotate90 = transforms.Compose([self.rotate90])

#         # rot180
#         rotate180 = transforms.Compose([self.rotate180])

#         # rot270
#         rotate270 = transforms.Compose([self.rotate270])

#         # flipH
#         flipH = transforms.Compose([self.flipH])

#         # flipH + rot90
#         flipH_rotate90 = transforms.Compose([self.flipH, self.rotate90])

#         # flipH + rot180
#         flipH_rotate180 = transforms.Compose([self.flipH, self.rotate180])

#         # flipH +rot270
#         flipH_rotate270 = transforms.Compose([self.flipH, self.rotate270])

#         # Add transformations to list
#         if self.use_transformations:
#             self.all_transformations.append(rotate90)
#             self.all_transformations.append(rotate180)
#             self.all_transformations.append(rotate270)
#             self.all_transformations.append(flipH)
#             self.all_transformations.append(flipH_rotate90)
#             self.all_transformations.append(flipH_rotate180)
#             self.all_transformations.append(flipH_rotate270)

#         self.nr_transformations = len(self.all_transformations)
#         self.images_per_transformation = self.nr_cubes * self.images_per_cube
#         self.length = self.images_per_transformation * (
#             self.nr_transformations + 1
#         )  # +1 for no transformation, i.e. original image

#         ### load data ###
#         self.cubes = []
#         self.slice_data = {}

#         cube_idx = 0
#         print("Loading data")
#         dataset_path = paths.get_full_dataset_path(self.redshift)
#         with h5py.File(dataset_path, "r") as f:
#             for seed in tqdm(self.seeds):
#                 # GR cube
#                 cube = (
#                     torch.tensor(f[f"gr_seed{seed:04d}"][()], dtype=torch.float32)
#                     if not lazy_load
#                     else f"gr_seed{seed:04d}"
#                 )
#                 label = torch.tensor([1.0 - target_noise], dtype=torch.float32)
#                 self.cubes.append({"cube": cube, "label": label})
#                 cube_idx += 1

#                 # Newton cube
#                 cube = (
#                     torch.tensor(
#                         f[f"newton_seed{seed:04d}"][()] * newton_augmentation,
#                         dtype=torch.float32,
#                     )
#                     if not lazy_load
#                     else f"newton_seed{seed:04d}"
#                 )
#                 label = torch.tensor([target_noise], dtype=torch.float32)
#                 self.cubes.append({"cube": cube, "label": label})
#                 cube_idx += 1

#         assert cube_idx == self.nr_cubes, "Cube index does not match number of cubes."

#         ### define slices ###
#         for slice_idx in range(self.images_per_cube):
#             slices = [slice(None)] * self.nr_axes
#             axis = slice_idx // self.images_per_axis
#             idx_on_slice = slice_idx % self.images_per_axis
#             slices[axis] = slice(
#                 idx_on_slice * self.stride, (idx_on_slice + 1) * self.stride
#             )
#             self.slice_data[slice_idx] = tuple(slices)
#         assert (
#             len(self.slice_data) == self.images_per_cube
#         ), "Slice index does not match number of slices."

#     def __len__(self) -> int:
#         return self.length

#     def __getitem__(self, idx) -> dict:
#         ### sub-indices ###
#         transform_idx = idx // self.images_per_transformation
#         seed_idx = idx % self.images_per_transformation
#         return (
#             self.get_sample(seed_idx, self.all_transformations[transform_idx - 1])
#             if transform_idx != 0
#             else self.get_sample(seed_idx)
#         )

#     def get_sample(self, seed_idx, transformation=None) -> dict:
#         cube_idx = seed_idx // self.images_per_cube
#         slice_idx = seed_idx % self.images_per_cube
#         cube = self.cubes[cube_idx]
#         sample_slice = self.slice_data[slice_idx]
#         if self.lazy_load:
#             with h5py.File(paths.get_full_dataset_path(self.redshift), "r") as f:
#                 image = torch.tensor(f[cube["cube"]][sample_slice], dtype=torch.float32)
#         else:
#             image = cube["cube"][sample_slice]
#         label = cube["label"]
#         sample = (
#             {
#                 "image": transformation(image.reshape(self.stride, 256, 256)),
#                 "label": label,
#             }
#             if transformation is not None
#             else {"image": image.reshape(self.stride, 256, 256), "label": label}
#         )
#         return sample


# def make_training_and_testing_data(
#     train_test_split,
#     train_test_seeds,
#     stride=1,
#     redshift=1.0,
#     random_seed=42,
#     transforms: bool = True,
#     newton_augmentation: float = 1.0,
#     lazy_load: bool = False,
#     target_noise: float = 0.0,
# ):
#     random.seed(random_seed)
#     random.shuffle(train_test_seeds)

#     array_length = len(train_test_seeds)
#     assert (
#         abs(sum(train_test_split) - 1.0) < 1e-6
#     ), "Train and test split does not sum to 1."
#     train_length = int(array_length * train_test_split[0])
#     test_length = int(array_length * train_test_split[1])
#     train_seeds = train_test_seeds[:train_length]
#     test_seeds = train_test_seeds[train_length:]

#     # Make datasets
#     print("Making datasets...")
#     print(f"Training set: {len(train_seeds)} seeds")
#     train_dataset = SlicedCubeDataset(
#         stride=stride,
#         redshift=redshift,
#         seeds=train_seeds,
#         use_transformations=transforms,
#         newton_augmentation=newton_augmentation,
#         lazy_load=lazy_load,
#         target_noise=target_noise,
#     )
#     print(f"Test set: {len(test_seeds)} seeds")
#     test_dataset = SlicedCubeDataset(
#         stride=stride,
#         redshift=redshift,
#         seeds=test_seeds,
#         use_transformations=transforms,
#         newton_augmentation=newton_augmentation,
#         lazy_load=lazy_load,
#         target_noise=target_noise,
#     )
#     return train_dataset, test_dataset
