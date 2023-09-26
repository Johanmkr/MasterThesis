import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import cube
from typing import Union


class CubeBispectrum:
    def __init__(self, cube_object:Union[cube.Cube, str], kwargs:dict={"threads": 1, "verbose": False}) -> None:
        if isinstance(cube_object, str):
            cube_object = cube.Cube(cube_object)
            self.cube = cube_object
            self.data = self.cube.data
            

if __name__=="__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    seed_nr = 1234
    gravity = "newton"
    redshift = 1
    path = datapath + f"seed{seed_nr:04d}/" + gravity + f"/{gravity}_{cube.redshift_to_snap[redshift]}_phi.h5"

    obj = cube.Cube(path)
    delta = obj()

    box_size = 256 #Mpc/h
    MAS = None #None, "CIC", "TSC", "PCS"
    axis = 0
    threads = 1
    verbose = True

    Pk = PKL.Pk(delta, box_size, axis, MAS, threads, verbose)

    k1D = Pk.k1D
    Pk1D = Pk.Pk1D
    Nmodeset1D = Pk.Nmodes1D