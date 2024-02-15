import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import pathlib as pl
import Pk_library as PKL
import os, sys
import h5py
from IPython import embed

# add path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from src.utils import paths

boxsize = 5120  # Mpc/h
Ngrid = 256
resolution = boxsize / Ngrid
kF = 2 * np.pi / boxsize
kN = np.pi / resolution

seed = 0
redshift = 1
k_range = np.geomspace(kF, kN, 10)
theta = np.array([2 / 3 * np.pi, 19 / 20 * np.pi])  # equilateral and squeezed

grpath = paths.get_cube_path(seed, "gr", redshift)
newtonpath = paths.get_cube_path(seed, "newton", redshift)
grpath_amp = paths.get_cube_path_amp(seed, "gr", redshift)
newtonpath_amp = paths.get_cube_path_amp(seed, "newton", redshift)

with h5py.File(grpath, "r") as f:
    grdata = np.array(f["data"][:], dtype=np.float32)
with h5py.File(newtonpath, "r") as f:
    newtondata = np.array(f["data"][:], dtype=np.float32)
with h5py.File(grpath_amp, "r") as f:
    grdata_amp = np.array(f["data"][:], dtype=np.float32)
with h5py.File(newtonpath_amp, "r") as f:
    newtondata_amp = np.array(f["data"][:], dtype=np.float32)

# embed()


B_gr = np.zeros((len(k_range), len(theta)))
# Q_gr = np.zeros(len(k_range))
B_newton = np.zeros((len(k_range), len(theta)))
# Q_newton = np.zeros(len(k_range))

B_gr_amp = np.zeros((len(k_range), len(theta)))
# Q_gr_amp = np.zeros(len(k_range))
B_newton_amp = np.zeros((len(k_range), len(theta)))
# Q_newton_amp = np.zeros(len(k_range))

P_gr = np.zeros((len(k_range), len(theta)))
P_newton = np.zeros((len(k_range), len(theta)))
P_gr_amp = np.zeros((len(k_range), len(theta)))
P_newton_amp = np.zeros((len(k_range), len(theta)))


for i, k in tqdm(enumerate(k_range)):
    print(k, k, theta)
    B_gr_calc = PKL.Bk(grdata, boxsize, k, k, theta, "CIC")
    B_newton_calc = PKL.Bk(newtondata, boxsize, k, k, theta, "CIC")
    B_gr_amp_calc = PKL.Bk(grdata_amp, boxsize, k, k, theta, "CIC")
    B_newton_amp_calc = PKL.Bk(newtondata_amp, boxsize, k, k, theta, "CIC")
    embed()

    for Bk_calc, Bk_fill in zip(
        [B_gr_calc, B_newton_calc, B_gr_amp_calc, B_newton_amp_calc],
        [B_gr, B_newton, B_gr_amp, B_newton_amp],
    ):
        Bk_fill[i] = Bk_calc.B

embed()
