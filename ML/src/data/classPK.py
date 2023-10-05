import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class_output = "/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/class_public/output/"

class ClassSpectra:
    def __init__(self, pk_path:str=class_output) -> None:
        self.pkPath = pk_path
        # try:
        #     self.pkData = np.loadtxt(pk_path)
        # except FileNotFoundError:
        #     print("File not found")
        #     exit(1)

        

    def __call__(self, redshift:float=0.0, gauge:str="synchronous", z2:bool=False) -> np.ndarray:
        z1path, z2path = self._gen_path(redshift, gauge)
        if z2:
            self.pkData = np.loadtxt(z2path)
        else:
            self.pkData = np.loadtxt(z1path)
        return self.pkData.T # Return transposed data with shape (2, length)
    
    def _gen_path(self, redshift:float, gauge:str) -> tuple:
        redshift_w_comma_separation = f"{redshift:.1f}".replace(".", ",")
        redshift_val = f"comparison_z_{redshift_w_comma_separation}_gauge_{gauge}"
        z1 = redshift_val + "00_z1_pk.dat"
        z2 = redshift_val + "00_z2_pk.dat"
        return (self.pkPath + z1, self.pkPath + z2)
    
if __name__=="__main__":
    pk = ClassSpectra()
    plt.loglog(*pk(gauge="synchronous"), label="Synchronous") 
    plt.loglog(*pk(gauge="newtonian"), label="Newtonian")
    plt.show()
    

