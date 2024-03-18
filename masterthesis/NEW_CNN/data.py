import numpy as np
import os, sys
import torch
import h5py
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize

from tqdm import tqdm


class ScaledData(Dataset):
    def __init__(
        self,
        seeds: np.ndarray = np.arange(0, 150),
        newton_augmentation: float = 1.0,
        A_s: float = 2.215e-9,
        train: bool = True,
        mean: float = None,
        variance: float = None,
    ):
        super(ScaledData, self).__init__()
        self.seeds = seeds
        self.newton_augmentation = newton_augmentation
        self.datapath = (
            "/mn/stornext/d10/data/johanmkr/simulations/prepared_data/"
            + f"scaled_data_A_s{A_s:.3e}.h5"
        )
        self.data = []
        self.npy_name = f"mean_var_A_s{A_s:.3e}.npy"

        cube_count = 0
        img_count = 0
        grlabel = torch.tensor([1.0], dtype=torch.float32)
        newtonlabel = torch.tensor([0.0], dtype=torch.float32)
        print("Loading training data...") if train else print("Loading test data...")
        for seed in tqdm(self.seeds):
            grseed = f"{seed}/gr"
            newtonseed = f"{seed}/newton"
            with h5py.File(self.datapath, "r") as f:
                grcube = torch.tensor(f[grseed][()], dtype=torch.float32)
                newtoncube = torch.tensor(f[newtonseed][()], dtype=torch.float32)
                newtoncube *= self.newton_augmentation
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
                    img_count += 3
            cube_count += 2

        # Calculate statistics
        if train:
            try:
                (load_mean, load_var) = np.load(self.npy_name)
                self.mean = load_mean
                self.variance = load_var
            except FileNotFoundError:
                print("Calculating mean and variance...")
                total_data = torch.cat([item["image"] for item in tqdm(self.data)])
                self.mean = total_data.mean()
                self.variance = total_data.var()
                np.save(self.npy_name, np.array([self.mean, self.variance]))
                del total_data
        else:
            self.mean = mean
            self.variance = variance

        self.norm = Normalize(mean=[self.mean], std=[self.variance**0.5])

        # Normalize data
        print("Normalizing data...")
        for item in tqdm(self.data):
            item["image"] = self.norm(item["image"])

        # Perform checks
        self.length = len(self.data)
        assert self.length == img_count
        assert cube_count == 2 * len(self.seeds)
        assert self.length == 2 * len(self.seeds) * 256 * 3
        print(f"Checked! Loaded {cube_count} cubes with {img_count} images")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


def create_data(
    train_seeds: np.ndarray,
    test_seeds: np.ndarray,
):
    train_dataset = ScaledData(seeds=train_seeds, train=True)
    test_dataset = ScaledData(
        seeds=test_seeds,
        train=False,
        mean=train_dataset.mean,
        variance=train_dataset.variance,
    )
    return train_dataset, test_dataset
