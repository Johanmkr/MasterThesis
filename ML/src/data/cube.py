import numpy as np
import os
import h5py

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


class Cube:
    def __init__(self, cube_path:str) -> None:
        """
            Initialise the Cube object. 
            args:
                cube_path: str
                    Path to the h5 file containing the cube.
        """
        self.cubePath = cube_path
        self.gr = "gr_snap" in self.cubePath
        self.seed = int(self.cubePath[-25:-21] if self.gr else self.cubePath[-33:-29]) # Will be 0000, 0001, etc.
        self.snapID = self.cubePath[-14:-7] # Will be snap000, snap001, etc.
        self.redshift = snap_to_redshift[self.snapID]

        self._read_cube()
    
    def _read_cube(self) -> None:
        """
            Read the cube from the h5 file.
        """
        h5File = h5py.File(self.cubePath, "r")
        self.h5Data = h5File["data"]
        self.data = self.h5Data[()]
        h5File.close()
    
    def __call__(self) -> np.ndarray:
        """
            Return the cube data.
        """
        return self.data
    
    def __getitem__(self, idx) -> np.ndarray:
        """
            Return the cube data for a given index/indices.
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