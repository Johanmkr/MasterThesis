import numpy as np
import os, sys
import torch
import h5py
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
import Pk_library as PKL

from tqdm import tqdm


boxsize = 5120 # Size of the box
Ngrid = 256 # Number of grid point in the box
resolution = boxsize/Ngrid # Resolution of the grid
kF = 2*np.pi/boxsize # Fundamental frequency
kN = np.pi/resolution # Nyquist frequency


from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

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
            "/mn/stornext/d10/data/johanmkr/simulations/prepared_data/" + f"pk_from_scaled_data.h5"
        )
        self.data = []
        # self.images = []
        # # self.npy_name = f"mean_var_A_s{A_s:.3e}.npy"

        # cube_count = 0
        # img_count = 0
        # grlabel = torch.tensor([1.0], dtype=torch.float32)
        # newtonlabel = torch.tensor([0.0], dtype=torch.float32)
        # for seed in tqdm(self.seeds, desc="Loading training data" if train else "Loading test data"):
        #     grseed = f"{seed}/gr"
        #     newtonseed = f"{seed}/newton"
        #     with h5py.File(self.datapath, "r") as f:
        #         grcube = np.array(f[grseed][()], dtype=np.float32)
        #         newtoncube = np.array(f[newtonseed][()], dtype=np.float32)
        #         newtoncube *= self.newton_augmentation
        #     for i in range(256):
        #         for cube, label in [(grcube, grlabel), (newtoncube, newtonlabel)]:
        #             self.images.append(
        #                 {"image": cube[i, :, :], "label": label}
        #             )
        #             self.images.append(
        #                 {"image": cube[:, i, :], "label": label}
        #             )
        #             self.images.append(
        #                 {"image": cube[:, :, i], "label": label}
        #             )
        #             img_count += 3
        #     cube_count += 2

        # Calculate Power spectra of the 2D images
        # for image in tqdm(self.images, desc="Calculating power spectra"):
        #     with suppress_stdout():
        #         Pk2D = PKL.Pk_plane(image["image"], BoxSize=5120, MAS="CIC", threads=32)
        #     k = Pk2D.k
        #     Pk = Pk2D.Pk
            
        #     # Omit k larger than Nyquist frequency and smaller than fundamental frequency
        #     mask = (k < kN) & (k > kF)
        #     k = k[mask]
        #     Pk = Pk[mask]
        #     item = {
        #         "Pk": torch.tensor(Pk, dtype=torch.float32),
        #         "k": torch.tensor(k, dtype=torch.float32),
        #         "label": image["label"],
        #     }
        
        # Load data
        with h5py.File(self.datapath, "r") as f:
            for seed in tqdm(self.seeds, desc="Loading training data" if train else "Loading test data"):
                grseed = f"{seed}/gr"
                newtonseed = f"{seed}/newton"
                gr_spectra = np.array(f[grseed][()], dtype=np.float32)
                newton_spectra = np.array(f[newtonseed][()], dtype=np.float32)
                assert len(gr_spectra) == len(newton_spectra) == 256*3
                for i in range(256*3):
                    item = {
                        "Pk": torch.tensor(gr_spectra[i], dtype=torch.float32),
                        "label": torch.tensor([1.0], dtype=torch.float32),
                    }
                    self.data.append(item)
                    item = {
                        "Pk": torch.tensor(newton_spectra[i], dtype=torch.float32),
                        "label": torch.tensor([0.0], dtype=torch.float32),
                    }
                    self.data.append(item)
        


        if train:
            # Calculate mean and variance
            try:
                (self.mean, self.variance) = np.load("mean_var.npy")
            except FileNotFoundError:
                total_data = np.concatenate([item["Pk"] for item in tqdm(self.data, desc="Concatenating data")])
                self.mean = total_data.mean()
                self.variance = total_data.var()
                np.save("mean_var.npy", np.array([self.mean, self.variance]))
                del total_data
        else:
            self.mean = mean
            self.variance = variance


        self.norm = lambda x: (x - self.mean) / self.variance**0.5

        # Normalize data
        for item in tqdm(self.data, desc="Normalizing data"):
            item["Pk"] = self.norm(item["Pk"])

        # Perform checks
        self.length = len(self.data)
        assert self.length == 2 * len(self.seeds) * 256 * 3
        print(f"Checked! Loaded {self.length} power spectra")

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

if __name__=="__main__":
    train_seeds = np.arange(0,2)
    test_seeds = np.arange(2, 3)
    train_dataset, test_dataset = create_data(train_seeds, test_seeds)
