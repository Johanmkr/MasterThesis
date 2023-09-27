import numpy as np
import os
import h5py
from typing import Union

# Used for testing only
from IPython import embed

snap_to_redshift = {
    "snap000": 20,
    "snap001": 15,
    "snap002": 10,
    "snap003": 5,
    "snap004": 1,
    "snap005": 0
}

redshift_to_snap = {
    20: "snap000",
    15: "snap001",
    10: "snap002",
    5: "snap003",
    1: "snap004",
    0: "snap005"
}

#TODO: I think this is wrong, but for testing only
# snap_to_redshift = {
#     "snap000": 0,
#     "snap001": 1,
#     "snap002": 5,
#     "snap003": 10,
#     "snap004": 15,
#     "snap005": 20
# }

# redshift_to_snap = {
#     0: "snap000",
#     1: "snap001",
#     5: "snap002",
#     10: "snap003",
#     15: "snap004",
#     20: "snap005"
# }


class Cube:
    def __init__(self, cube_path:str, normalise:bool=False) -> None:
        """
            Initialise the Cube object. 
            Args:
                cube_path (str): Path to the h5 file containing the cube.
        """
        self.cubePath = cube_path
        self.gr = "gr_snap" in self.cubePath
        self.seed = int(self.cubePath[-25:-21] if self.gr else self.cubePath[-33:-29]) # Will be 0000, 0001, etc.
        self.snapID = self.cubePath[-14:-7] # Will be snap000, snap001, etc.
        self.redshift = snap_to_redshift[self.snapID]

        self._read_cube()
        if normalise:
            self._normalise_cube()

        self.boxsize = 5120 #Mpc/h
        self.Ngrid = self.data.shape[0]
        self.resolution = self.boxsize / self.Ngrid
        self.kF = 2*np.pi / self.boxsize
        self.kN = np.pi / self.resolution
    
    def _read_cube(self) -> None:
        """
            Read the cube from the h5 file.
        """
        h5File = h5py.File(self.cubePath, "r")
        self.h5Data = h5File["data"]
        self.data = self.h5Data[()]
        h5File.close()

    def _normalise_cube(self) -> None:
        """
            Normalise the cube.
        """
        self.cubeMean = np.mean(self.data)
        self.cubeStd = np.std(self.data)
        self.data = (self.data - self.cubeMean)/self.cubeStd

    def get_gradient(self, dim:int=0) -> None:
        """
            Get the gradient of the cube.
            Args:   
                dim (int): The dimension to take the gradient in.
        """
        self.gradient = np.gradient(self.data)[dim]
        return self.gradient

    ###TODO: fix the below functino to take the dimension into account. 
    def get_laplacian(self) -> None:
        """
            Get the laplacian of the cube.
        """
        self.laplacian = np.gradient(np.gradient(self.data))
        return self.laplacian
    
    def __call__(self) -> np.ndarray:
        """
            Return the cube data.
        """
        return self.data
    
    def __getitem__(self, idx:Union[int, tuple, slice, list]) -> np.ndarray:
        """
            Return the cube data for a given index/indices.
            Args:
                idx (int, list, tuple, slice): The index/indices to return.
        """
        return self.data[idx]
    
    def __str__(self) -> str:
        """
            Return a string representation of the Cube object.
        """
        RT = "\nCube Information\n----------------------------\n"
        RT += f"Path: {'/'.join(self.cubePath.split('/')[-3:])}\n"
        RT += f"Shape: {self.data.shape}\n"
        RT += f"Seed: {self.seed}\n"
        RT += f"Redshift: {self.redshift}\n"
        RT += f"Gravity: {'gr' if self.gr else 'newton'}\n"
        RT += "----------------------------\n"
        return RT
    
if __name__=="__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    seed_nr = 1234
    gravity = "gr"
    redshift = 20
    path = datapath + f"seed{seed_nr}/" + gravity + f"/{gravity}_{redshift_to_snap[redshift]}_phi.h5"

    obj = Cube(path)
    print(obj)
    # embed()