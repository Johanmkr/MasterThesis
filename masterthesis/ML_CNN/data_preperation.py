import numpy as np
import pandas as pd
import os, sys
import torch
from scipy.interpolate import interp1d
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
import h5py
from tqdm import tqdm


# paths
background_info_path = (
    "/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/class_public/gf_output/"
)
sync_file = background_info_path + "default00_background.dat"
simulation_path = "/mn/stornext/d10/data/johanmkr/simulations/"

# Generate growth factor spline
header_names = [
    "z",
    "t",
    "tau",
    "H",
    "comv_dist",
    "ang_diam_dist",
    "lum_dist",
    "comv_snd_hrz",
    "rho_g",
    "rho_b",
    "rho_cdm",
    "tho_lambda",
    "rho_ur",
    "rho_crit",
    "rho_tot",
    "p_tot",
    "p_tot_prime",
    "D",
    "f",
]
sync_data = np.loadtxt(sync_file)
sync_df = pd.DataFrame(sync_data, columns=header_names)
Dz = interp1d(sync_df["z"], sync_df["D"], kind="cubic")


# Conversions
snap_to_redshift = {
    "snap000": 20,
    "snap001": 15,
    "snap002": 10,
    "snap003": 5,
    "snap004": 1,
    "snap005": 0,
}

redshift_to_snap = {
    20: "snap000",
    15: "snap001",
    10: "snap002",
    5: "snap003",
    1: "snap004",
    0: "snap005",
}


# function to create the difference-maps
def create_diff_map(image_first, image_last, z_first=10, z_last=1):
    return (image_first / Dz(z_first)) - (image_last / Dz(z_last))


class NormalDataset(Dataset):
    def __init__(
        self,
        seeds: np.ndarray,
        A_s: float = 2.1e-9,
        z: float = 1.0,
        newton_augmentation: float = 1.0,
    ):
        super(NormalDataset, self).__init__()

        # Generate path to data and create self variables
        self.datapath = simulation_path + f"{A_s:.2e}/"
        self.newton_augmentation = newton_augmentation
        self.data = []  # Empty list to store data

        # Making sure the count is correct
        cube_count = 0
        img_count = 0
        grlabel = torch.tensor([1.0], dtype=torch.float32)
        newtonlabel = torch.tensor([0.0], dtype=torch.float32)

        # Set tensor for mean
        self.total_sum = torch.zeros((256, 256, 256), dtype=torch.float32)

        # Loop through seeds
        for seed in tqdm(seeds):
            seedpath = self.datapath + f"seed{seed:04d}/"
            grpath = seedpath + f"gr/gr_{redshift_to_snap[int(z)]}_phi.h5"
            newtonpath = seedpath + f"newton/newton_{redshift_to_snap[int(z)]}_phi.h5"

            # Open files
            with h5py.File(grpath, "r") as grf:
                with h5py.File(newtonpath, "r") as newtonf:
                    # Load the cubes
                    grcube = torch.tensor(grf["data"][()], dtype=torch.float32)
                    self.total_sum += grcube
                    newtoncube = torch.tensor(newtonf["data"][()], dtype=torch.float32)
                    self.total_sum += newtoncube

                    newtoncube *= self.newton_augmentation

            # Loop through dimensions and extract images
            for i in range(256):
                for cube, label in [(grcube, grlabel), (newtoncube, newtonlabel)]:
                    self.data.append(
                        {"image": cube[i, :, :].unsqueeze(0), "label": label}
                    )
                    self.data.append(
                        {"image": cube[:, i, :].unsqueeze(0), "label": label}
                    )
                    self.data.append(
                        {"image": cube[:, :, i].unsqueeze(0), "label": label}
                    )
                    img += 3
            cube_count += 2

        # Check
        self.length = len(self.data)
        assert self.length == img_count
        assert cube_count == 2 * len(seeds)
        assert self.length == 2 * len(seeds) * 256 * 3
        print(f"Loaded {cube_count} cubes with {img_count} images")

        # Calculate mean and std
        self.mean = self.total_sum / cube_count
        self.mean = self.mean.mean()
        self.std = self.total_sum.std() / cube_count

        # Create normalization function
        self.normalize = Normalize(mean=[self.mean], std=[self.std])

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            item = self.data[idx]
            item["image"] = self.normalize(item["image"])
            return item


class DifferenceDataset(Dataset):
    def __init__(
        self,
        seeds: np.ndarray,
        A_s: float = 2.1e-9,
        z_first: float = 10.0,
        z_last: float = 1.0,
        newton_augmentation: float = 1.0,
    ):
        super(NormalDataset, self).__init__()

        # Generate path to data and create self variables
        self.datapath = simulation_path + f"{A_s:.2e}/"
        self.newton_augmentation = newton_augmentation
        self.data = []  # Empty list to store data

        # Making sure the count is correct
        cube_count = 0
        img_count = 0
        grlabel = torch.tensor([1.0], dtype=torch.float32)
        newtonlabel = torch.tensor([0.0], dtype=torch.float32)

        # Set tensor for mean
        self.total_sum = torch.zeros((256, 256, 256), dtype=torch.float32)

        # Loop through seeds
        for seed in tqdm(seeds):
            seedpath = self.datapath + f"seed{seed:04d}/"
            grpath_first = seedpath + f"gr/gr_{redshift_to_snap[int(z_first)]}_phi.h5"
            grpath_last = seedpath + f"gr/gr_{redshift_to_snap[int(z_last)]}_phi.h5"

            newtonpath_first = (
                seedpath + f"newton/newton_{redshift_to_snap[int(z_first)]}_phi.h5"
            )
            newtonpath_last = (
                seedpath + f"newton/newton_{redshift_to_snap[int(z_last)]}_phi.h5"
            )

            # Open files
            # GR
            with h5py.File(grpath_first, "r") as grf_first:
                # Load the cubes
                grcube_first = torch.tensor(grf_first["data"][()], dtype=torch.float32)
            with h5py.File(grpath_last, "r") as grf_last:
                grcube_last = torch.tensor(grf_last["data"][()], dtype=torch.float32)
            grcube = create_diff_map(grcube_first, grcube_last)
            self.total_sum += grcube
            cube_coutn += 1
            # Newton
            with h5py.File(newtonpath_first, "r") as newtonf_first:
                newtoncube_first = torch.tensor(
                    newtonf_first["data"][()], dtype=torch.float32
                )
            with h5py.File(newtonpath_last, "r") as newtonf_last:
                newtoncube_last = torch.tensor(
                    newtonf_last["data"][()], dtype=torch.float32
                )
            newtoncube = create_diff_map(newtoncube_first, newtoncube_last)
            self.total_sum += newtoncube
            newtoncube *= self.newton_augmentation
            cube_count += 1

            # Loop through dimensions and extract images
            for i in range(256):
                for cube, label in [(grcube, grlabel), (newtoncube, newtonlabel)]:
                    self.data.append(
                        {"image": cube[i, :, :].unsqueeze(0), "label": label}
                    )
                    self.data.append(
                        {"image": cube[:, i, :].unsqueeze(0), "label": label}
                    )
                    self.data.append(
                        {"image": cube[:, :, i].unsqueeze(0), "label": label}
                    )
                    img += 3

        # Check
        self.length = len(self.data)
        assert self.length == img_count
        assert cube_count == 2 * len(seeds)
        assert self.length == 2 * len(seeds) * 256 * 3
        print(f"Loaded {cube_count} cubes with {img_count} images")

        # Calculate mean and std
        self.mean = self.total_sum / cube_count
        self.mean = self.mean.mean()
        self.std = self.total_sum.std() / cube_count

        # Create normalization function
        self.normalize = Normalize(mean=[self.mean], std=[self.std])

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            item = self.data[idx]
            item["image"] = self.normalize(item["image"])
            return item
