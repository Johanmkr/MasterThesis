# script to check if simulations are run successfully
import os


def check_simulations(seed, A_s, gravity):
    # Check gravity
    if gravity not in ["gr", "newton"]:
        raise ValueError("gravity must be 'gr' or 'newton'")

    # Check seed
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if seed < 0 or seed >= 250:
        raise ValueError("seed must be between 0 and 2000")

    # Correct format
    seed = str(seed).zfill(4)
    A_s = f"{A_s:.3e}"

    # Output path
    output_path = (
        f"/mn/stornext/d10/data/johanmkr/simulations/{A_s}/seed{seed}/{gravity}/"
    )

    # Check output path
