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
    if seed < 0 or seed >= 2000:
        raise ValueError("seed must be between 0 and 2000")

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
    temp_init_file = f"{os.getcwd()}/{gravity}_temp.ini"
    _perform_changes(infile, temp_init_file, changes)


def _cleanup():
    print("Cleaning up")
    os.remove(f"{os.getcwd()}/gr_temp.ini")
    os.remove(f"{os.getcwd()}/newton_temp.ini")


def execute_simulation(
    seed,
    A_s,
):
    # Make temporary init files
    try:
        generate_temp_init(seed, A_s, "gr")
        generate_temp_init(seed, A_s, "newton")

        # Run simulations
        os.system(f"{os.getcwd()}/simulate.sh")

        # Clean up
        _cleanup()
    except RuntimeError:
        pass


if __name__ == "__main__":
    for seed in np.arange(0, 20, 1):
        for A_s in [2.215e-9, 2.215e-8, 2.215e-7, 2.215e-6, 2.215e-5, 2.215e-4]:
            execute_simulation(int(seed), A_s)
