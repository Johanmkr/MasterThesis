import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class_dat_file = "/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/class_public/output/comparison_NBS00_pk.dat"

class ClassSpectra:
    def __init__(self, pk_path:str=class_dat_file) -> None:
        self.pkPath = pk_path
        try:
            self.pkData = np.loadtxt(pk_path)
        except FileNotFoundError:
            print("File not found")
            exit(1)

    def __call__(self) -> np.ndarray:
        return self.pkData.T # Return transposed data with shape (2, length)
    
if __name__=="__main__":
    pk = ClassPK()
    plt.loglog(*pk())
    plt.show()
    

