import numpy as np
import matplotlib.pyplot as plt
import bispectrum as bs
import cube
import pandas as pd

seeds = np.arange(0,2000,50)
redshifts = [0, 1, 5, 15]

datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"

test_seed = 1234
output_folder = "pre_computed_bispectra"

for seed in seeds:
    for redshift in redshifts:
        print(f"\n\n Computing bispectra for seed {seed:04d} at redshift {redshift:04d}\n\n")

        grBispectrum = bs.CubeBispectrum(datapath + f"seed{seed:04d}/gr/gr_{cube.redshift_to_snap[redshift]}_phi.h5")
        newtonBispectrum = bs.CubeBispectrum(datapath + f"seed{seed:04d}/newton/newton_{cube.redshift_to_snap[redshift]}_phi.h5")

        k_range = np.geomspace(grBispectrum.kF, 1e-2, 1000)

        grdBkeq = grBispectrum.equilateral_bispectrum(k_range, {"threads": 24})
        ndBkeq = newtonBispectrum.equilateral_bispectrum(k_range, {"threads": 24})
        grdBksq = grBispectrum.squeezed_bispectrum(k_range, {"threads": 24})
        ndBksq = newtonBispectrum.squeezed_bispectrum(k_range, {"threads": 24})

        grdBkeq.to_csv(f"{output_folder}/seed{seed:04d}_gr_equilateral_rs{redshift:04d}.csv")
        ndBkeq.to_csv(f"{output_folder}/seed{seed:04d}_newton_equilateral_rs{redshift:04d}.csv")
        grdBksq.to_csv(f"{output_folder}/seed{seed:04d}_gr_squeezed_rs{redshift:04d}.csv")
        ndBksq.to_csv(f"{output_folder}/seed{seed:04d}_newton_squeezed_rs{redshift:04d}.csv")

# grBispectrum = bs.CubeBispectrum(datapath + f"seed{test_seed:04d}/gr/gr_{cube.redshift_to_snap[redshifts[0]]}_phi.h5")
# newtonBispectrum = bs.CubeBispectrum(datapath + f"seed{test_seed:04d}/newton/newton_{cube.redshift_to_snap[redshifts[0]]}_phi.h5")

# k_range = np.geomspace(grBispectrum.kF, 1e-2, 15)

# grdBk = grBispectrum.equilateral_bispectrum(k_range)
# ndBk = newtonBispectrum.equilateral_bispectrum(k_range)

# grdBk.to_csv(f"{output_folder}/seed{test_seed:04d}_gr_equilateral_rs{redshifts[0]:04d}.csv")
# ndBk.to_csv(f"{output_folder}/seed{test_seed:04d}_newton_equilateral_rs{redshifts[0]:04d}.csv")
