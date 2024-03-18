import numpy as np
import sys, os
import pandas as pd
import h5py
from tqdm import trange, tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scaling_task import scaling


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
# Paths
datapath = "/mn/stornext/d10/data/johanmkr/simulations/"
path = (
    lambda A_s, seed, gravity, z: datapath
    + f"{A_s:.3e}/seed{seed.zfill(4)}/{gravity.lower()}/{gravity.lower()}_{redshift_to_snap[z]}_phi.h5"
)
output_path = "/mn/stornext/d10/data/johanmkr/simulations/prepared_data/"


def create_scaled_dataset(A_s):
    # Set scaler for difference dataset
    scaler = scaling.CubeScaler(A_s)

    # Open h5 file to write to
    with h5py.File(output_path + f"scaled_data_A_s{A_s:.3e}.h5", "w") as f:

        # Iterate over seeds and gravities
        for seed in tqdm(np.arange(0, 300)):
            for gravity in ["gr", "newton"]:

                # Read GR and Newton data
                with h5py.File(path(A_s, str(seed), gravity, 10), "r") as z10:
                    z10data = np.array(z10["data"][:], dtype=np.float32)
                with h5py.File(path(A_s, str(seed), gravity, 1), "r") as z1:
                    z1data = np.array(z1["data"][:], dtype=np.float32)

                # Scale and save difference dataset
                diffcube = scaler.scale_10_1(z10data, z1data)
                f.create_dataset(
                    f"{seed}/{gravity}",
                    data=diffcube,
                    dtype=np.float32,
                    compression="gzip",
                )


if __name__ == "__main__":
    create_scaled_dataset(2.215e-9)
