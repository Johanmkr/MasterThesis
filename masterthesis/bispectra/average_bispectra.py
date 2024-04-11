import numpy as np
import pandas as pd
import os
from tqdm import trange


datapath = "/mn/stornext/d10/data/johanmkr/simulations/bispectra_analysis/"
outputpath = (
    "/mn/stornext/d10/data/johanmkr/simulations/bispectra_analysis/average_bispectra/"
)

# check if outputpath exists
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

z10path = (
    lambda seed, A_s, gravity: datapath
    + f"z_10/seed{str(seed).zfill(4)}/{A_s:.3e}_{gravity.lower()}.pkl"
)
z1path = (
    lambda seed, A_s, gravity: datapath
    + f"z_1/seed{str(seed).zfill(4)}/{A_s:.3e}_{gravity.lower()}.pkl"
)

scaledpath = (
    lambda seed, A_s, gravity: datapath
    + f"scaled/seed{str(seed).zfill(4)}/SCALED_{A_s:.3e}_{gravity.lower()}.pkl"
)

pathfinder = {
    "z10": z10path,
    "z1": z1path,
    "scaled": scaledpath,
}


def calculate_and_save_average(gravity, kind, A_s=2.215e-9, max_seed=250, KSTEPS=75):
    """
    Calculate and save the average bispectrum for a given gravity model and A_s
    """
    assert kind in ["z10", "z1", "scaled"], "kind must be 'z10', 'z1' or 'scaled'"

    # Create empty arrays
    B_eq = np.zeros((max_seed, KSTEPS))
    B_sq = np.zeros((max_seed, KSTEPS))
    B_st = np.zeros((max_seed, KSTEPS))
    Q_eq = np.zeros((max_seed, KSTEPS))
    Q_sq = np.zeros((max_seed, KSTEPS))
    Q_st = np.zeros((max_seed, KSTEPS))
    Pk = np.zeros((max_seed, KSTEPS))

    # Loop over seeds
    for seed in trange(max_seed):
        # Load dataframes
        frame = pd.read_pickle(pathfinder[kind](seed, A_s, gravity))

        # Add to arrays
        B_eq[seed] = frame["B_eq"]
        B_sq[seed] = frame["B_sq"]
        B_st[seed] = frame["B_st"]
        Q_eq[seed] = frame["Q_eq"]
        Q_sq[seed] = frame["Q_sq"]
        Q_st[seed] = frame["Q_st"]
        Pk[seed] = frame["Pk"]

    # Calculate average
    B_eq_avg = np.mean(B_eq, axis=0)
    B_sq_avg = np.mean(B_sq, axis=0)
    B_st_avg = np.mean(B_st, axis=0)
    Q_eq_avg = np.mean(Q_eq, axis=0)
    Q_sq_avg = np.mean(Q_sq, axis=0)
    Q_st_avg = np.mean(Q_st, axis=0)
    Pk_avg = np.mean(Pk, axis=0)

    # Calculate standard deviation
    B_eq_std = np.std(B_eq, axis=0)
    B_sq_std = np.std(B_sq, axis=0)
    B_st_std = np.std(B_st, axis=0)
    Q_eq_std = np.std(Q_eq, axis=0)
    Q_sq_std = np.std(Q_sq, axis=0)
    Q_st_std = np.std(Q_st, axis=0)
    Pk_std = np.std(Pk, axis=0)

    # Create and save dataframe
    avg_frame = pd.DataFrame(
        {
            "k": frame["k"],
            "B_eq_avg": B_eq_avg,
            "B_eq_std": B_eq_std,
            "B_sq_avg": B_sq_avg,
            "B_sq_std": B_sq_std,
            "B_st_avg": B_st_avg,
            "B_st_std": B_st_std,
            "Q_eq_avg": Q_eq_avg,
            "Q_eq_std": Q_eq_std,
            "Q_sq_avg": Q_sq_avg,
            "Q_sq_std": Q_sq_std,
            "Q_st_avg": Q_st_avg,
            "Q_st_std": Q_st_std,
            "Pk_avg": Pk_avg,
            "Pk_std": Pk_std,
            "kind": kind,
            "max_seed": max_seed,
        }
    )
    savename = f"B_avg_{kind}_{A_s:.2e}_{gravity.lower()}.pkl"

    savepath = outputpath + savename
    avg_frame.to_pickle(savepath)
    print(f"Saved {savepath}")


if __name__ == "__main__":
    kinds = ["z10", "z1", "scaled"]
    gravities = ["GR", "Newton"]
    for kind in kinds:
        for gravity in gravities:
            calculate_and_save_average(gravity, kind, max_seed=300, KSTEPS=75)
