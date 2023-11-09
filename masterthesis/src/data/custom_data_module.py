from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import numpy as np

from . import custom_dataset


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        stride: int = 2,
        redshifts: int | float | list | tuple = 1.0,
        seeds: np.ndarray | int | float | list | tuple = np.arange(0, 2000, 1),
        axes: int | list | tuple = [0, 1, 2],
        transform: callable = None,
        additional_info=False,
        train_test_val_split: tuple = (0.6, 0.2, 0.2),
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.redshifts = redshifts
        self.seeds = seeds
        self.axes = axes
        self.stride = stride
        self.transform = transform
        self.additional_info = additional_info
        assert sum(train_test_val_split) == 1, "train_val_split must sum to 1"
        self.train_test_val_split = train_test_val_split
        self.split_seed = split_seed

        self.prepare_data()

    def prepare_data(self):
        self.dataset = custom_dataset.CustomDataset(
            stride=self.stride,
            redshifts=self.redshifts,
            seeds=self.seeds,
            axes=self.axes,
            transform=self.transform,
            additional_info=self.additional_info,
        )

    def setup(self, stage):
        train_size = int(self.train_test_val_split[0] * len(self.dataset))
        test_size = int(self.train_test_val_split[1] * len(self.dataset))
        val_size = len(self.dataset) - train_size - test_size
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, test_size, val_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def __str__(self):
        return self.dataset.__str__()

    def print_image(self, idx):
        return self.dataset.print_image(idx)
