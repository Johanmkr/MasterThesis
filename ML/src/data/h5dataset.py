import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5cube 
import h5collection

# Used for testing only
from IPython import embed

class DatasetOfCubes(Dataset):
    def __init__(self, data_dir:str, stride:int=1, Ngrid:int=128, boxsize:float=5120.0, multiple_directories:bool=False, exceptions:str=None) -> None:
        self.dataDir = data_dir
        self.stride = stride
        self.Ngrid = Ngrid
        self.boxsize = boxsize
        self.multipleDirectories = multiple_directories
        self.exceptions = exceptions
        self.directories = []
        self.collections = []

        if self.multipleDirectories:
            self._find_directories()
        else:
            self.directories.append(self.dataDir)
        
        #   Initialise the collections
        for directory in self.directories:
            newton_path = directory + "/newton/"
            gr_path = directory + "/gr/"
            self.collections.append(h5collection.Collection(newton_path))
            self.collections.append(h5collection.Collection(gr_path))

        #   Find length
        self.nrDims = 3 #   Nr of dimensions
        self.nrCollections = len(self.collections)  #   Collection objects
        self.listOfCollectionLengths = [obj.nrRedshifts for obj in self.collections]
        self.nrCubes = sum(self.listOfCollectionLengths)
        self.nrSlicesPerDim = self.Ngrid // self.stride + bool(Ngrid % stride) #  Number of slices including the smaller remainder slice if present

        self.totalLenght = self.nrSlicesPerDim * self.nrDims * self.nrCubes




    def __len__(self):
        return self.totalLenght

    def __getitem__(self, long_idx):
        pass

    def _find_directories(self) -> None:
        directory_entries = os.listdir(self.dataDir)
        if self.exceptions is not None:
            for entry in directory_entries not in self.exceptions:
                self.directories.append(entry)
            self.directories.sort()
        else:
            self.directories = directory_entries.sort()

        
if __name__=="__main__":
    output_path = os.path.abspath("").replace("Summer-Sandbox23/ML/src/data", "NbodySimulation/gevolution-1.2/output")
    run_type = "/intermediate"

    path = output_path + run_type
    DS = DatasetOfCubes(path)
    embed()



