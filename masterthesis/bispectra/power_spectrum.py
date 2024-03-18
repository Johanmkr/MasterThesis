import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from IPython import embed

power_spectrum_path = "/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/class_public/non_lin_output/"

# Matter power spectrum in synchronous gauge
z0_pk_path = power_spectrum_path + "def_z1_pk.dat"
z0_pk_nonlin_path = power_spectrum_path + "def_z1_pk_nl.dat"
z1_pk_path = power_spectrum_path + "def_z2_pk.dat"
z1_pk_nonlin_path = power_spectrum_path + "def_z2_pk_nl.dat"
z10_pk_path = power_spectrum_path + "def_z3_pk.dat"
z10_pk_path_nonlin = power_spectrum_path + "def_z3_pk_nl.dat"

# Read power spectrum data
labels = ["k", "Pk"]
z0_pk = pd.DataFrame(np.loadtxt(z0_pk_path), columns=labels)
z0_pk_nl = pd.DataFrame(np.loadtxt(z0_pk_nonlin_path), columns=labels)
z1_pk = pd.DataFrame(np.loadtxt(z1_pk_path), columns=labels)
z1_pk_nl = pd.DataFrame(np.loadtxt(z1_pk_nonlin_path), columns=labels)
z10_pk = pd.DataFrame(np.loadtxt(z10_pk_path), columns=labels)
z10_pk_nl = pd.DataFrame(np.loadtxt(z10_pk_path_nonlin), columns=labels)

# Make interpolations
z0_pk_interp = interp1d(
    z0_pk["k"], z0_pk["Pk"], kind="cubic", bounds_error=False, fill_value="extrapolate"
)
z0_pk_nl_interp = interp1d(
    z0_pk_nl["k"],
    z0_pk_nl["Pk"],
    kind="cubic",
    bounds_error=False,
    fill_value="extrapolate",
)
z1_pk_interp = interp1d(
    z1_pk["k"], z1_pk["Pk"], kind="cubic", bounds_error=False, fill_value="extrapolate"
)
z1_pk_nl_interp = interp1d(
    z1_pk_nl["k"],
    z1_pk_nl["Pk"],
    kind="cubic",
    bounds_error=False,
    fill_value="extrapolate",
)
z10_pk_interp = interp1d(
    z10_pk["k"],
    z10_pk["Pk"],
    kind="cubic",
    bounds_error=False,
    fill_value="extrapolate",
)
z10_pk_nl_interp = interp1d(
    z10_pk_nl["k"],
    z10_pk_nl["Pk"],
    kind="cubic",
    bounds_error=False,
    fill_value="extrapolate",
)


def get_Pk(z, k, nl=False):
    z = int(z)
    assert z in [0, 1, 10], "z must be 0, 1 or 10"
    if z == 0:
        if nl:
            return z0_pk_nl_interp(k)
        else:
            return z0_pk_interp(k)
    elif z == 1:
        if nl:
            return z1_pk_nl_interp(k)
        else:
            return z1_pk_interp(k)
    elif z == 10:
        if nl:
            return z10_pk_nl_interp(k)
        else:
            return z10_pk_interp(k)


if __name__ == "__main__":
    k = np.logspace(-3, 1, 1000)
    plt.plot(k, Pk(0, k), label="z=0")
    plt.plot(k, Pk(1, k), label="z=1")
    plt.plot(k, Pk(10, k), label="z=10")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
    # embed()
