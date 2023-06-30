import numpy as np
import os
import h5py

# Used for testing only
from IPython import embed

class Cube:
    def __init__(self, cubePath:str, redshift:str or int=None) -> None:
        self.cubePath = cubePath
        self.redshift = redshift if redshift is not None else "N/A"
        self._readCube()
    
    def _readCube(self) -> None:
        h5File = h5py.File(self.cubePath, "r")
        self.h5Data = h5File["data"]
        self.data = self.h5Data[()]
        h5File.close()
    
    def __call__(self) -> np.ndarray:
        return self.data
    
    def __getitem__(self, idx) -> np.ndarray:
        return self.data[idx]
    
    def __str__(self) -> str:
        RT = "\nh5Cube Information\n----------------------------\n"
        RT += f"Path: {'/'.join(self.cubePath.split('/')[-4:])}\n"
        RT += f"Shape: {self.data.shape}\n"
        RT += f"Redshift: {self.redshift}\n"
        RT += "----------------------------\n"
        return RT
    
if __name__=="__main__":
    output_path = os.path.abspath("").replace("Summer-Sandbox23/ML/src/data", "NbodySimulation/gevolution-1.2/output")
    run_type = "/intermediate"
    gravity = "/gr"
    filename = "/lcdm_snap000_phi.h5"
    path = output_path+run_type+gravity+filename

    embed()