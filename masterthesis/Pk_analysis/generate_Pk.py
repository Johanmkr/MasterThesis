import numpy as np
import os, sys
import h5py
from tqdm import trange, tqdm
import Pk_library as PKL

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


boxsize = 5120  # Size of the box
Ngrid = 256  # Number of grid point in the box
resolution = boxsize / Ngrid  # Resolution of the grid
kF = 2 * np.pi / boxsize  # Fundamental frequency
kN = np.pi / resolution  # Nyquist frequency

datapath = (
    "/mn/stornext/d10/data/johanmkr/simulations/prepared_data/"
    + f"scaled_data_A_s{2.215e-9:.3e}.h5"
)

total_seeds = np.arange(0, 300)

output_path = (
    "/mn/stornext/d10/data/johanmkr/simulations/prepared_data/pk_from_scaled_data.h5"
)

output_cube_path = (
    "/mn/stornext/d10/data/johanmkr/simulations/prepared_data/pk_from_scaled_cubes.h5"
)


def get_masked_data(Pk2D):
    k = Pk2D.k
    mask = (k > kF) & (k < kN)
    return Pk2D.Pk[mask]


def get_masked_data_from_3D_pk(Pk3D):
    k = Pk3D.k3D
    mask = (k > kF) & (k < kN)
    return Pk3D.Pk[mask, 0]  # Returns monopole contribution


def calculate_Pks():
    with h5py.File(output_path, "w") as f:
        for seed in tqdm(total_seeds, desc="Calculating Pks"):
            grseed = f"{seed}/gr"
            newtonseed = f"{seed}/newton"
            with h5py.File(datapath, "r") as lf:
                grcube = np.array(lf[grseed][()], dtype=np.float32)
                newtoncube = np.array(lf[newtonseed][()], dtype=np.float32)
            grPKs = []
            newtonPKs = []
            for i in range(256):
                for cube, label in [(grcube, "gr"), (newtoncube, "newton")]:
                    # Get images
                    im1 = cube[i, :, :]
                    im2 = cube[:, i, :]
                    im3 = cube[:, :, i]

                    # Calculate Pks

                    with suppress_stdout():
                        Pk2D_1 = PKL.Pk_plane(im1, BoxSize=5120, MAS="CIC", threads=8)
                        Pk2D_2 = PKL.Pk_plane(im2, BoxSize=5120, MAS="CIC", threads=8)
                        Pk2D_3 = PKL.Pk_plane(im3, BoxSize=5120, MAS="CIC", threads=8)

                    if label == "gr":
                        grPKs.append(get_masked_data(Pk2D_1))
                        grPKs.append(get_masked_data(Pk2D_2))
                        grPKs.append(get_masked_data(Pk2D_3))
                    else:
                        newtonPKs.append(get_masked_data(Pk2D_1))
                        newtonPKs.append(get_masked_data(Pk2D_2))
                        newtonPKs.append(get_masked_data(Pk2D_3))
            f.create_dataset(
                f"{seed}/gr", data=np.array(grPKs), dtype=np.float32, compression="gzip"
            )
            f.create_dataset(
                f"{seed}/newton",
                data=np.array(newtonPKs),
                dtype=np.float32,
                compression="gzip",
            )


def calculate_Pks_from_cubes():
    with h5py.File(output_cube_path, "w") as f:
        for seed in tqdm(total_seeds, desc="Calculate 3D Pks"):
            grseed = f"{seed}/gr"
            newtonseed = f"{seed}/newton"
            with h5py.File(datapath, "r") as lf:
                grcube = np.array(lf[grseed][()], dtype=np.float32)
                newtoncube = np.array(lf[newtonseed][()], dtype=np.float32)
            with suppress_stdout():
                grPk3D = PKL.Pk(grcube, BoxSize=5120, axis=0, MAS="CIC", threads=8)
                newtonPk3D = PKL.Pk(
                    newtoncube, BoxSize=5120, axis=0, MAS="CIC", threads=8
                )
            f.create_dataset(
                f"{seed}/gr",
                data=np.array(get_masked_data_from_3D_pk(grPk3D)),
                dtype=np.float32,
                compression="gzip",
            )
            f.create_dataset(
                f"{seed}/newton",
                data=np.array(get_masked_data_from_3D_pk(newtonPk3D)),
                dtype=np.float32,
                compression="gzip",
            )


if __name__ == "__main__":
    # calculate_Pks()
    calculate_Pks_from_cubes()
