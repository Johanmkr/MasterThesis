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
from . import classPK


class AnalyticalBispectrum:
    """
    Class for calculating the analytical bispectrum. The class takes a range of wavenumbers and a redshift as input. The class then calculates the equilateral and squeezed bispectrum analytically and stores the results in the instance variables:
    - B_equilateral
    - B_squeezed.

    Args:
        k_range (np.array): array of wavenumbers
        z (float, optional): redshift. Defaults to 1.0.
    """

    def __init__(self, k_range: np.array, z: float = 1.0) -> None:
        self.k_range = k_range
        self.B_equilateral = np.zeros(len(k_range))
        self.B_squeezed = np.zeros(len(k_range))

        class_obj = classPK.ClassSpectra(z)
        self.phi_spline = self._splined_PS(
            class_obj.phi_pk, bounds_error=False, fill_value="extrapolate"
        )
        self._analytical_equilateral_bispectrum(self.phi_spline)
        self._analytical_squeezed_bispectrum(self.phi_spline)

    def _F2_kernel(self, k1: float, k2: float, angle: float) -> float:
        return (
            5.0 / 7
            + 1.0 / 2 * (k1 / k2 + k2 / k1) * np.cos(angle)
            + 2.0 / 7 * (np.cos(angle)) ** 2
        )

    def _full_F2_output(self, k: float, theta_12: float) -> tuple:
        """The F2 kernel for equilateral and squeezed triangles where k1=k2"""
        alpha = np.pi - theta_12
        k1 = k
        k2 = k
        k3 = np.sqrt(k1**2 + k2**2 - 2 * k1 * k2 * np.cos(alpha))

        # Find beta and gamma
        beta = np.arcsin(k1 / k3 * np.sin(alpha))
        gamma = np.arcsin(k2 / k3 * np.sin(alpha))

        # Check that the angles add up to pi
        assert np.isclose(beta + gamma + alpha, np.pi).all()

        # Find remaining thetas
        theta_23 = np.pi - beta
        theta_13 = np.pi - gamma

        # Permutations
        F12 = self._F2_kernel(k1, k2, theta_12)
        F23 = self._F2_kernel(k2, k3, theta_23)
        F31 = self._F2_kernel(k3, k1, theta_13)

        return F12, F23, F31, k3

    def _splined_PS(self, PS: pd.DataFrame, kind: str = "cubic", **kwargs):
        """Returns a spline interpolation of the power spectra"""
        f = interp1d(PS.k, PS.pk, kind=kind, **kwargs)
        return f

    def _analytical_equilateral_bispectrum(self, ps_function: callable) -> None:
        """Calculates the equilateral bispectrum analytically"""
        F12, F23, F31, k3 = self._full_F2_output(self.k_range, theta_12=2 * np.pi / 3)
        self.B_equilateral = (
            2 * F12 * ps_function(self.k_range) * ps_function(self.k_range)
            + 2 * F23 * ps_function(self.k_range) * ps_function(k3)
            + 2 * F31 * ps_function(self.k_range) * ps_function(k3)
        )
        # for i, k in enumerate(self.k_range):
        #     F12, F23, F31, k3 = self._full_F2_output(k, np.pi / 3)
        #     self.B_equilateral[i] = (
        #         2 * F12 * ps_function(k) * ps_function(k)
        #         + 2 * F23 * ps_function(k) * ps_function(k3)
        #         + 2 * F31 * ps_function(k) * ps_function(k3)
        #     )

    def _analytical_squeezed_bispectrum(self, ps_function: callable) -> None:
        """Calculates the squeezed bispectrum analytically"""
        F12, F23, F31, k3 = self._full_F2_output(self.k_range, theta_12=19 * np.pi / 20)
        self.B_squeezed = (
            2 * F12 * ps_function(self.k_range) * ps_function(self.k_range)
            + 2 * F23 * ps_function(self.k_range) * ps_function(k3)
            + 2 * F31 * ps_function(self.k_range) * ps_function(k3)
        )
        # for i, k in enumerate(self.k_range):
        #     F12, F23, F31, k3 = self._full_F2_output(k, 19 * np.pi / 20)
        #     self.B_squeezed[i] = (
        #         2 * F12 * ps_function(k) * ps_function(k)
        #         + 2 * F23 * ps_function(k) * ps_function(k3)
        #         + 2 * F31 * ps_function(k) * ps_function(k3)
        #     )

    # def _get_PS_spline()

    def get_custom_bispectrum(
        self, k_range: np.ndarray | float, theta_range: np.ndarray | float
    ) -> np.ndarray:
        """Calculates the bispectrum for a custom k and theta range"""
        if isinstance(k_range, (int, float)):
            k_range = np.array([k_range])
        if isinstance(theta_range, (int, float)):
            theta_range = np.array([theta_range])
        B = np.zeros((len(k_range), len(theta_range)))
        for j, theta in enumerate(theta_range):
            F12, F23, F31, k3 = self._full_F2_output(k_range, theta)
            B[:, j] = (
                2 * F12 * self.phi_spline(k_range) * self.phi_spline(k_range)
                + 2 * F23 * self.phi_spline(k_range) * self.phi_spline(k3)
                + 2 * F31 * self.phi_spline(k_range) * self.phi_spline(k3)
            )
        return B


if __name__ == "__main__":
    print("Bispectrum module")
