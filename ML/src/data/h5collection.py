import numpy as np
import os
import h5py
import h5cube

# Used for testing only
from IPython import embed

class Collection:
    def __init__(self, dir_path:str) -> None:
        self.dirPath = dir_path
        self.cubes = []

        #   Run initialisation procedures
        self._find_files()

    def _find_files(self) -> None:
        self.allFiles = os.listdir(self.dirPath)
        self.configDict = {}
        self.initFile = [self.dirPath + name for name in self.allFiles if ".ini" in name][0]
        with open(self.initFile, "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                self.configDict[key.strip()] = value.strip()
        self.haveIni = True
        self.redshifts = [int(num) for num in self.configDict["snapshot redshifts"].split(",")]
        self.redshifts.sort(reverse=True) # redshifts decrease
        # embed()
        self.nrRedshifts = len(self.redshifts)

        self.h5Files = ([self.dirPath + name for name in self.allFiles if ".h5" in name])
        self.h5Files.sort()

        for i in range(self.nrRedshifts):
            self.cubes.append(h5cube.Cube(self.h5Files[i], redshift=int(self.redshifts[i]), gr=self.configDict["gravity theory"]=="GR"))
    
    def __getitem__(self, idx:int or tuple/list) -> np.ndarray or h5cube.Cube:
        if isinstance(idx, int):
            return self.cubes[idx]
        else:
            return self.cubes[idx[0]][idx[1:]]
        
    def __str__(self) -> str:
        RT = "\nCollection configuration\n----------------------------\n"
        for key, value in self.configDict.items():
            RT += f"{key}: {value}\n"
        RT += "----------------------------\n"
        RT += "\n\nCollection information\n----------------------------\n"
        RT += f"Path: {'/'.join(self.dirPath.split('/')[-4:])}\n"
        RT += f"Nr. of cubes: {self.nrRedshifts}\n"
        RT += "----------------------------\n"
        return RT
    
if __name__=="__main__":
    output_path = os.path.abspath("").replace("Summer-Sandbox23/ML/src/data", "NbodySimulation/gevolution-1.2/output")
    run_type = "/intermediate"
    gravity = "/gr"
    path = output_path + run_type + gravity + "/"
    obj = Collection(path)
    embed()
    
