# Global imports
import numpy as np
import os
import torch
from torch.utils.data import Dataset

# Local imports
from . import cube
from . import transforms
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

    def __len__(self):
        return self.nr_images

    def __getitem__(self, idx):
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


if __name__ == "__main__":
    pass
