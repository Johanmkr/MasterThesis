import numpy as np
import os, sys
import h5py
from tqdm import tqdm, trange

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from src.utils import paths
import cube


def prepare_cube_and_move_to_correct_directory(seed, gravity, redshift, statistics):
    """
    This function takes a cube and moves it to the correct directory
    """
    mean, std, variance = statistics
    # Read cube
    cube_path = paths.get_cube_path(seed, gravity, redshift)
    with h5py.File(cube_path, "r") as f:
        cube = np.array(f["data"][()], dtype=np.float32)

    # normalise cube
    cube = (cube - mean) / std

    save_path = paths.get_prepared_cube_path(seed, gravity, redshift)
    with h5py.File(save_path, "w") as f:
        f.create_dataset("data", data=cube, dtype=np.float32)


def find_mean_and_std(redshift):
    try:
        mean_std_var = np.load(
            f"{os.path.dirname(__file__)}/redshifts_[{redshift:.1f}]_mean_std_var.npy",
            "r",
        )
        mean = mean_std_var[0]
        std = mean_std_var[1]
        variance = mean_std_var[2]
        return (mean, std, variance)
    except FileNotFoundError:
        print("File with mean, standard deviation and variance not found")


def make_many_small_h5_files(redshift):
    statistics = find_mean_and_std(redshift)
    for seed in trange(2000):
        for gravity in ["GR", "Newton"]:
            # Prepare cube
            prepare_cube_and_move_to_correct_directory(
                seed, gravity, redshift, statistics
            )


def make_one_big_h5_file(redshift):
    mean, std, variance = find_mean_and_std(redshift)
    filename = f"data_z{redshift:.0f}.h5"
    save_path = paths.datapath / f"data_z{redshift:.0f}" / filename

    # create large file
    with h5py.File(save_path, "w") as f:
        for seed in trange(2000):
            for gravity in ["GR", "Newton"]:
                cube_path = paths.get_cube_path(seed, gravity, redshift)
                with h5py.File(cube_path, "r") as f2:
                    cube = np.array(f2["data"][()], dtype=np.float32)
                cube = (cube - mean) / std
                f.create_dataset(
                    f"{gravity.lower()}_seed{seed:04d}", data=cube, dtype=np.float32
                )
    print(f"Successfully created {save_path}")


if __name__ == "__main__":
    # Get command line argument for redshift
    redshift = float(input("Redshift: "))
    make_big = (
        True if input("Make one big file? (y/n): ").lower() in ["y", "yes"] else False
    )
    make_small = (
        True
        if input("Make many small files? (y/n): ").lower() in ["y", "yes"]
        else False
    )

    if make_big:
        make_one_big_h5_file(redshift)
    if make_small:
        make_many_small_h5_files(redshift)
