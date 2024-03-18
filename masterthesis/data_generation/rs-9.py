import numpy as np
import sys, os


def _perform_changes(infile, outfile, changes):
    with open(infile, "r") as f:
        lines = f.readlines()
    with open(outfile, "w") as f:
        for line in lines:
            for key, value in changes.items():
                if key in line:
                    line = line.replace(key, value)
            f.write(line)


# Function to create the new temporary init file given the new parameters
def generate_temp_init(seed, A_s, gravity):
    # Perform checks
    # Check gravity
    if gravity not in ["gr", "newton"]:
        raise ValueError("gravity must be 'gr' or 'newton'")

    # Check seed
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if seed < 0 or seed >= 250:
        raise ValueError("seed must be between 0 and 250")

    # Correct format
    seed = str(seed).zfill(4)
    A_s = f"{A_s:.3e}"

    # Check output path
    output_path = (
        f"/mn/stornext/d10/data/johanmkr/simulations/{A_s}/seed{seed}/{gravity}/"
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # check if output path is empty
    if os.listdir(output_path):
        print(f"Output path {output_path} is not empty")
        raise RuntimeError("Output path is not empty")

    # make tag
    tag = f"seed{seed}_A_s{A_s}_gravity{gravity}"

    # Define the changes to be made
    changes = {
        "seed =": f"seed = {seed}",
        "A_s =": f"A_s = {A_s}",
        "output path =": f"output path = {output_path}",
    }

    # Make temporary file
    infile = f"initialisations/{gravity}.ini"
    temp_init_file = f"{os.getcwd()}/{gravity}_{seed}_{A_s}_temp.ini"
    _perform_changes(infile, temp_init_file, changes)


def _cleanup(seed, A_s):
    print("Cleaning up")
    seed = str(seed).zfill(4)
    A_s = f"{A_s:.3e}"
    os.remove(f"{os.getcwd()}/gr_{seed}_{A_s}_temp.ini")
    os.remove(f"{os.getcwd()}/newton_{seed}_{A_s}_temp.ini")


def execute_simulation(
    seed,
    A_s,
):
    # Make temporary init files
    try:
        generate_temp_init(seed, A_s, "gr")
        generate_temp_init(seed, A_s, "newton")
        tmp_file_gr = f"{os.getcwd()}/gr_{str(seed).zfill(4)}_{f'{A_s:.3e}'}_temp.ini"
        tmp_file_newton = (
            f"{os.getcwd()}/newton_{str(seed).zfill(4)}_{f'{A_s:.3e}'}_temp.ini"
        )
        # Run simulations
        os.system(f"{os.getcwd()}/simulate.sh {tmp_file_gr}")
        os.system(f"{os.getcwd()}/simulate.sh {tmp_file_newton}")

        # Clean up
        _cleanup(seed, A_s)
    except RuntimeError:
        pass


if __name__ == "__main__":
    for seed in np.arange(0, 250, 1):
        for A_s in [2.215e-9]:
            execute_simulation(int(seed), A_s)
    # execute_simulation(0, 2.215e-9)
