import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import pathlib as pl
import Pk_library as PKL
import os, sys
import h5py
from IPython import embed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scaling_task import scaling

# Global variables
boxsize = 5120  # Mpc/h
Ngrid = 256
resolution = boxsize / Ngrid
kF = 2 * np.pi / boxsize
kN = np.pi / resolution

# For calculating bispectra
THREADS = 128
KSTEPS = 75

# Convert to right units
# h = 0.67556
# kF /= h  # Convert to h/Mpc
# kN /= h  # Convert to h/Mpc

# Paths
datapath = "/mn/stornext/d10/data/johanmkr/simulations/"

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


# equilateral:
# k1 = k2 = k3
# mu = 0.5, t = 1

# Squeezed
# k1 = k2 >> k3
# mu = 1, t = 1

# Stretched
# k1 >> k2 = k3
# mu = 1, t = 0.5


def _get_bispectra(phi, k1, mu, t):
    # Check values
    assert k1 > 0
    assert t >= 0.5
    assert 2 * mu * t >= 1
    assert mu <= 1

    k2 = k1 * t
    theta = np.arccos(
        -mu
    )  # must be negative since pylians defines the angle in an opposite way from me
    B = PKL.Bk(phi, boxsize, k1, k2, np.array([theta]), "CIC", threads=THREADS)
    bispectra = np.zeros(2)
    power_spectrum = np.array([B.Pk[0]])  # Power spectrum for k1 mode
    bispectra[0] = B.B
    bispectra[1] = B.Q
    # return_array[2] = B.Pk
    return bispectra, power_spectrum


def calculate_statistics_for_cube(
    phi: np.ndarray,
    nr_ksteps: int = KSTEPS,
    save_path: str = None,
):

    # Generate k_range
    k_range = np.geomspace(kF, kN, nr_ksteps)  # TODO CHECK THIS RANGE

    B_equilateral = np.zeros((len(k_range), 2))
    B_squeezed = np.zeros((len(k_range), 2))
    B_stretched = np.zeros((len(k_range), 2))
    Pk = np.zeros(len(k_range))

    for i, k in enumerate(k_range):
        print(f"Calculating bispectra for k={k:.2e} ({i+1}/{len(k_range)})")
        # Stretched:
        B_stretched[i], _ = _get_bispectra(phi, k, mu=0.99, t=0.51)
        # Squeezed:
        B_squeezed[i], _ = _get_bispectra(phi, k, mu=0.99, t=1)
        # Equilateral:
        B_equilateral[i], Pk[i] = _get_bispectra(phi, k, mu=0.5, t=1)

    # Create empty dataframe
    df = pd.DataFrame()
    df["k"] = k_range
    df["B_eq"] = B_equilateral[:, 0]
    df["Q_eq"] = B_equilateral[:, 1]
    df["B_sq"] = B_squeezed[:, 0]
    df["Q_sq"] = B_squeezed[:, 1]
    df["B_st"] = B_stretched[:, 0]
    df["Q_st"] = B_stretched[:, 1]
    df["Pk"] = Pk

    if save_path is not None:
        df.to_pickle(save_path)
    return df


def locate_cube_and_calculate(seed, gravity, A_s, scaler):
    A_s = f"{A_s:.3e}"
    if gravity not in ["gr", "newton"]:
        raise ValueError("Gravity must be either 'gr' or 'newton'")
    assert isinstance(seed, int)
    seed = str(seed).zfill(4)

    # Locate cubes
    z10_cube_path = (
        datapath
        + f"{A_s}/seed{seed}/{gravity}/{gravity}_{redshift_to_snap[int(10)]}_phi.h5"
    )
    z1_cube_path = (
        datapath
        + f"{A_s}/seed{seed}/{gravity}/{gravity}_{redshift_to_snap[int(1)]}_phi.h5"
    )

    # Check if cube exists
    for cube_path in [z10_cube_path, z1_cube_path]:
        if not pl.Path(cube_path).exists():
            print(f"\n\n\n\nCube not found: {cube_path}")
            return
        else:
            print(f"\n\n\n\nCube found: {cube_path}")

    # Load cubes
    print(f"\nLoading cubes...\n")
    with h5py.File(z10_cube_path, "r") as file:
        z10_data = np.array(file["data"][:], dtype=np.float32)
    with h5py.File(z1_cube_path, "r") as file:
        z1_data = np.array(file["data"][:], dtype=np.float32)

    scaled_phi = scaler.scale_10_1(z10_data, z1_data)

    output_savepath = datapath + f"bispectra_analysis/scaled/seed{seed}/"

    # Check if output path exists
    if not pl.Path(output_savepath).exists():
        os.makedirs(output_savepath)

    savename = f"SCALED_{A_s}_{gravity}.pkl"
    savepath = output_savepath + savename

    # Check if file already exists
    if pl.Path(savepath).exists():
        print(f"File already exists: {savepath}")
        return
    else:
        print(
            f"\n\nCalculating for\nSeed: {seed}\nA_s: {A_s}\nGravity: {gravity}\nSaving to {savepath}\n\n"
        )
        calculate_statistics_for_cube(scaled_phi, save_path=savepath)


if __name__ == "__main__":
    # locate_cube_and_calculate(0, "gr", 20, 2.215e-9)
    seed_ranges = [np.arange(0, 100, 1), np.arange(100, 200, 1), np.arange(200, 300)]
    seeds = seed_ranges[
        int(input("Enter seed range (0:[0,100) - 1:[100,200) - 2:[200,300) :"))
    ]
    A_s = [2.215e-9]
    gravities = ["gr", "newton"]
    A = A_s[0]
    for seed in seeds:
        scaler = scaling.CubeScaler(A_s=A)
        for gravity in gravities:
            locate_cube_and_calculate(int(seed), gravity, A, scaler)
