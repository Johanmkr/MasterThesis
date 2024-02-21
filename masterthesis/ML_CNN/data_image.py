import numpy as np
import os, sys
import torch
import h5py
from torch.utils.data.dataset import Dataset

from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(
        self,
        seeds: np.ndarray = np.arange(0, 150),
        newton_augmentation: float = 1.0,
        datapath: str = None,
    ):
        super(ImageDataset, self).__init__()
        self.seeds = seeds
        self.newton_augmentation = newton_augmentation
        if datapath is None:
            raise ValueError("datapath is None")
        self.datapath = datapath
        self.data = []

        cube_count = 0
        img_count = 0
        grlabel = torch.tensor([1.0], dtype=torch.float32)
        newtonlabel = torch.tensor([0.0], dtype=torch.float32)
        for seed in tqdm(self.seeds):
            grseed = f"gr_seed{seed:04d}"
            newtonseed = f"newton_seed{seed:04d}"
            with h5py.File(datapath, "r") as f:
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

        # Check
        self.length = len(self.data)
        assert self.length == img_count
        assert cube_count == 2 * len(self.seeds)
        assert self.length == 2 * len(self.seeds) * 256 * 3
        print(f"Loaded {cube_count} cubes with {img_count} images")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


def make_training_and_testing_data(
    train_seeds,
    test_seeds,
    newton_augmentation: float = 1.0,
    datapath: str = None,
):
    print("Making datasets...")
    print(f"Training set: {train_seeds} seeds")
    train_dataset = ImageDataset(
        seeds=train_seeds,
        newton_augmentation=newton_augmentation,
        datapath=datapath,
    )
    print(f"Testing set: {test_seeds} seeds")
    test_dataset = ImageDataset(
        seeds=test_seeds,
        newton_augmentation=newton_augmentation,
        datapath=datapath,
    )
    return train_dataset, test_dataset
