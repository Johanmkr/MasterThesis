import numpy as np

from power_spectrum import get_Pk

# Constants
omega_m = 0.022032 + 0.12038  # omega_b + omega_c


def _F2_kernel(k1, k2, psi):
    return 5.0 / 7 - psi / 2 * (k1 / k2 + k2 / k1) + 2.0 / 7 * psi**2


def C(k, a):
    H0k = 1 / (2997.13 * k)  # H0 in Mpc^-1
    return 3 * omega_m * H0k**2 / (2 * a)


def get_Bk(k1, mu, t, z, nl=False):
    # Calculate variables
    a = 1 / (1 + z)
    k2 = t * k1
    k3 = (k1 * k1 + k2 * k2 - 2 * k1 * k2 * mu) ** 0.5
    psi1 = mu
    psi2 = (k2 * k2 + k3 * k3 - k1 * k1) / (2 * k2 * k3)
    psi3 = (k3 * k3 + k1 * k1 - k2 * k2) / (2 * k3 * k1)

    assert np.isclose(
        np.arccos(psi1) + np.arccos(psi2) + np.arccos(psi3), np.pi
    ).all(), "Sum of angles must be pi"

    # Calculate F2 kernels
    F12 = _F2_kernel(k1, k2, psi1)
    F23 = _F2_kernel(k2, k3, psi2)
    F31 = _F2_kernel(k3, k1, psi3)

    # Get matter powerspectra in synchronous gauge
    Pk1 = get_Pk(z, k1, nl)
    Pk2 = get_Pk(z, k2, nl)
    Pk3 = get_Pk(z, k3, nl)

    # Calculate matter power spectrum
    B_delta = 2 * Pk1 * Pk2 * F12 + 2 * Pk2 * Pk3 * F23 + 2 * Pk3 * Pk1 * F31

    # Convert to B_phi
    B_phi = B_delta * C(k1, a) * C(k2, a) * C(k3, a)

    return B_phi


if __name__ == "__main__":
    # Plot bispectrum
    import matplotlib.pyplot as plt

    k = np.logspace(-3, 1, 1000)
    # equilateral:
    # k1 = k2 = k3
    # mu = 0.5, t = 1
    B_eq_z1 = get_Bk(k, mu=0.5, t=1, z=1)
    B_eq_z10 = get_Bk(k, mu=0.5, t=1, z=10)
    B_eq_z1_nl = get_Bk(k, mu=0.5, t=1, z=1, nl=True)
    B_eq_z10_nl = get_Bk(k, mu=0.5, t=1, z=10, nl=True)

    # Squeezed
    # k1 = k2 >> k3
    # mu = 1, t = 1
    B_sq_z1 = get_Bk(k, mu=0.99, t=0.99, z=1)
    B_sq_z10 = get_Bk(k, mu=0.99, t=0.99, z=10)
    B_sq_z1_nl = get_Bk(k, mu=0.99, t=0.99, z=1, nl=True)
    B_sq_z10_nl = get_Bk(k, mu=0.99, t=0.99, z=10, nl=True)

    # Stretched
    # k1 >> k2 = k3
    # mu = 1, t = 0.5
    B_st_z1 = get_Bk(k, mu=0.99, t=0.51, z=1)
    B_st_z10 = get_Bk(k, mu=0.99, t=0.51, z=10)
    B_st_z1_nl = get_Bk(k, mu=0.99, t=0.51, z=1, nl=True)
    B_st_z10_nl = get_Bk(k, mu=0.99, t=0.51, z=10, nl=True)

    # Plot

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax[0].loglog(k, abs(B_eq_z1), color="red", label="z=1")
    ax[0].loglog(k, abs(B_eq_z10), color="blue", label="z=10")
    ax[0].loglog(k, abs(B_eq_z1_nl), color="red", linestyle="--")
    ax[0].loglog(k, abs(B_eq_z10_nl), color="blue", linestyle="--")
    ax[0].set_title("Equilateral")

    ax[1].loglog(k, abs(B_sq_z1), color="red", label="z=1")
    ax[1].loglog(k, abs(B_sq_z10), color="blue", label="z=10")
    ax[1].loglog(k, abs(B_sq_z1_nl), color="red", linestyle="--")
    ax[1].loglog(k, abs(B_sq_z10_nl), color="blue", linestyle="--")
    ax[1].set_title("Squeezed")

    ax[2].loglog(k, abs(B_st_z1), color="red", label="z=1")
    ax[2].loglog(k, abs(B_st_z10), color="blue", label="z=10")
    ax[2].loglog(k, abs(B_st_z1_nl), color="red", linestyle="--")
    ax[2].loglog(k, abs(B_st_z10_nl), color="blue", linestyle="--")
    ax[2].set_title("Stretched")

    plt.legend()
    plt.show()
