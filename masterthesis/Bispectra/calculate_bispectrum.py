import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import pathlib as pl
import Pk_library as PKL
import os, sys
import h5py
from IPython import embed

# add path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from src.utils import paths

boxsize = 5120  # Mpc/h
Ngrid = 256
resolution = boxsize / Ngrid
kF = 2 * np.pi / boxsize
kN = np.pi / resolution

# equilateral:
# k1 = k2 = k3
# mu = 0.5, t = 1

# Squeezed
# k1 = k2 >> k3
# mu = 1, t = 1

# Stretched
# k1 >> k2 = k3
# mu = 1, t = 0.5


def get_bispectra(phi, k1, mu, t, threads: int = 64, Pk: bool = False):
    # Check values
    assert k1 > 0
    assert t >= 0.5
    assert 2 * mu * t >= 1
    assert mu <= 1

    k2 = k1 * t
    theta = np.arccos(
        -mu
    )  # must be negative since pylians want the angle between k1 and k2 to be
    B = PKL.Bk(phi, boxsize, k1, k2, np.array([theta]), "CIC", threads)
    bispectra = np.zeros(2)
    power_spectrum = np.array([B.Pk[0]])
    bispectra[0] = B.B
    bispectra[1] = B.Q
    # return_array[2] = B.Pk
    return bispectra, power_spectrum


def get_bispectra_for_k_range(phi, nr_k: float = 10, threads=64):
    k = np.geomspace(kF, kN, nr_k)
    equilateral = np.zeros((len(k), 3))
    squeezed = np.zeros((len(k), 3))
    stretched = np.zeros((len(k), 3))
    for i, k1 in enumerate(k):
        equilateral[i] = get_bispectra(phi, k1, mu=0.5, t=1, threads=threads)
        squeezed[i] = get_bispectra(phi, k1, mu=0.99, t=0.99, threads=threads)
        stretched[i] = get_bispectra(phi, k1, mu=0.99, t=0.51, threads=threads)
    return k, equilateral, squeezed, stretched


def calculate_statistics_for_cube(
    file_path: str,
    nr_ksteps: int = 50,
    save_path: str = None,
):
    # Load cube data
    with h5py.File(file_path, "r") as file:
        data = np.array(file["data"][:], dtype=np.float32)

    # Generate k_range
    k_range = np.geomspace(kF, kN, nr_ksteps)

    B_equilateral = np.zeros((len(k_range), 2))
    B_squeezed = np.zeros((len(k_range), 2))
    B_stretched = np.zeros((len(k_range), 2))
    Pk = np.zeros(len(k_range))

    for i, k in enumerate(k_range):
        # Stretched:
        B_stretched[i], _ = get_bispectra(data, k, mu=0.99, t=0.51)
        # Squeezed:
        B_squeezed[i], _ = get_bispectra(data, k, mu=0.99, t=0.99)
        # Equilateral:
        B_equilateral[i], Pk[i] = get_bispectra(data, k, mu=0.5, t=1, Pk=True)

    # Create empty dataframe
    df = pd.DataFrame()
    df["k"] = k_range
    df["B_equilateral"] = B_equilateral[:, 0]
    df["Q_equilateral"] = B_equilateral[:, 1]
    df["B_squeezed"] = B_squeezed[:, 0]
    df["Q_squeezed"] = B_squeezed[:, 1]
    df["B_stretched"] = B_stretched[:, 0]
    df["Q_stretched"] = B_stretched[:, 1]
    df["Pk"] = Pk

    if save_path is not None:
        df.to_pickle(save_path)
    return df


if __name__ == "__main__":
    print("Calculating bispectra for a cube")
    calculate_statistics_for_cube(
        paths.get_cube_path(seed=2, gravity="gr", redshift=1),
        save_path="test_bispectra/n_gr.pkl",
    )
    calculate_statistics_for_cube(
        paths.get_cube_path(seed=2, gravity="newton", redshift=1),
        save_path="test_bispectra/n_newton.pkl",
    )

    print("Calculating bispectra for a cube")
    calculate_statistics_for_cube(
        paths.get_cube_path_even_amp(seed=2, gravity="gr", redshift=1),
        save_path="test_bispectra/Ae-7_gr.pkl",
    )
    calculate_statistics_for_cube(
        paths.get_cube_path_even_amp(seed=2, gravity="newton", redshift=1),
        save_path="test_bispectra/Ae-7_newton.pkl",
    )

    print("Calculating bispectra for a cube")
    calculate_statistics_for_cube(
        paths.get_cube_path_amp(seed=2, gravity="gr", redshift=1),
        save_path="test_bispectra/Ae-6_gr.pkl",
    )
    calculate_statistics_for_cube(
        paths.get_cube_path_amp(seed=2, gravity="newton", redshift=1),
        save_path="test_bispectra/Ae-6_newton.pkl",
    )
