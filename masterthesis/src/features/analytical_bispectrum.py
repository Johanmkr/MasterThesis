import sys
import os

# add path to parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

# Global imports
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Local imports
from . import bispectrum as bs
from . import powerspectra as ps


class AnalyticalBispectrum:
    def __init__(self, k_range: np.array) -> None:
        self.k_range = k_range

    def F2_kernel(self, k1: float, k2: float, angle: float) -> float:
        return (
            5.0 / 7
            + 1.0 / 2 * (k1 / k2 + k2 / k1) * np.cos(angle)
            + 2.0 / 7 * (np.cos(angle)) ** 2
        )

    def full_F2_output(self, k: float, theta_12: float) -> tuple:
        """The F2 kernel for equilateral and squeezed triangles where k1=k2"""
        alpha = np.pi - theta_12
        k1 = k
        k2 = k
        k3 = np.sqrt(k1**2 + k2**2 - 2 * k1 * k2 * np.cos(alpha))

        # Find beta and gamma
        beta = np.arcsin(k1 / k3 * np.sin(alpha))
        gamma = np.arcsin(k2 / k3 * np.sin(alpha))

        # Check that the angles add up to pi
        assert np.isclose(beta + gamma + alpha, np.pi)

        # Find remaining thetas
        theta_23 = np.pi - beta
        theta_13 = np.pi - gamma

        # Permutations
        F12 = self.F2_kernel(k1, k2, theta_12)
        F23 = self.F2_kernel(k2, k3, theta_23)
        F31 = self.F2_kernel(k3, k1, theta_13)

        return F12, F23, F31, k3

    def Splined_PS(sefl, PS: pd.DataFrame, kind: str = "cubic", **kwargs):
        """Returns a spline interpolation of the power spectra"""
        f = interp1d(PS.k, PS.pk, kind=kind, **kwargs)
        return f
