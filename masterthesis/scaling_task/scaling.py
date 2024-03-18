import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

A_s_paths = {
    2.215e-9: "e-9_00_background.dat",
    2.215e-8: "e-8_00_background.dat",
    2.215e-7: "e-7_00_background.dat",
    2.215e-6: "e-6_00_background.dat",
    2.215e-5: "e-5_00_background.dat",
}
header_names = [
    "z",
    "t",
    "tau",
    "H",
    "comv_dist",
    "ang_diam_dist",
    "lum_dist",
    "comv_snd_hrz",
    "rho_g",
    "rho_b",
    "rho_cdm",
    "tho_lambda",
    "rho_ur",
    "rho_crit",
    "rho_tot",
    "p_tot",
    "p_tot_prime",
    "D",
    "f",
]

class CubeScaler:
    def __init__(self, A_s=2.215e-9):
        # Find file name for the given A_s
        background_info_path = "/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/class_public/gf_output/"
        gf_file = background_info_path + A_s_paths[A_s]

        # Read file
        gf_frame = pd.DataFrame(np.loadtxt(gf_file), columns=header_names)

        # Make interpolation of growth factor
        Dz = interp1d(gf_frame["z"], gf_frame["D"], kind="cubic")

        # Make function for the potential growth factor
        self.Dz_phi = lambda z: Dz(z) * (1 + z)
        self.D10_phi = self.Dz_phi(10)
        self.D1_phi = self.Dz_phi(1)

    def scale_10_1(self, cube_z10, cube_z1):
        return (cube_z10 / self.D10_phi) - (cube_z1 / self.D1_phi)
