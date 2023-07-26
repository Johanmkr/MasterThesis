import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5cube 
import h5collection

# Used for testing only
from IPython import embed

class DatasetOfCubes(Dataset):
    def __init__(self, data_dir:str, stride:int=1, Ngrid:int=128, boxsize:float=5120.0, multiple_directories:bool=False, exceptions:str=None, verbose:bool=False) -> None:
        self.dataDir = data_dir
        self.stride = stride
        self.Ngrid = Ngrid
        self.boxsize = boxsize
        self.multipleDirectories = multiple_directories
        self.exceptions = exceptions
        self.verbose = verbose
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
            if self.verbose:
                print(f"Initialised     {directory}")

        #   Find lengths
        self.nrDims = 3 #   Nr of dimensions
        self.nrCollections = len(self.collections)  #   Collection objects
        self.listOfCollectionLengths = [obj.nrRedshifts for obj in self.collections]
        self.lengthOfCollection = self.nrCollections # assuming all collections have the same length
        self.nrSlicesPerDim = self.Ngrid // self.stride + bool(Ngrid % stride) #  Number of slices including the smaller remainder slice if present

        self.nrCubes = sum(self.listOfCollectionLengths)
        self.nrSlicesPerCube = self.nrSlicesPerDim * self.nrDims
        self.totalLength = self.nrSlicesPerCube * self.nrCubes

        # Useful stuff for finding the correct indices -> assumes cubes to be of equal size
        self.nrSlicesPerCollection = np.array(self.listOfCollectionLengths) * self.nrSlicesPerCube
        # self.nrSlicesPerCollection = int(self.totalLength / self.lengthOfCollection)

        self.splitterArray = np.cumsum([self.nrSlicesPerCollection])[:-1]
        self.indicesPerCollection = np.split(np.arange(self.totalLength), self.splitterArray)
        # embed()




    def __len__(self):
        return self.totalLength

    def __getitem__(self, long_idx):
        # If we assume all collections to be of equal size (of cubes of equal size)
        collection_idx = long_idx // self.nrSlicesPerCollection     #   Index of collection
        cube_idx = long_idx % self.nrSlicesPerCollection            #   Index of cube within collection 
        idx_on_one_cube = long_idx % self.nrCubes                   #   Index on the cube
        dim_idx = idx_on_one_cube // self.Ngrid                     #   Index of dimension
        slice_idx = idx_on_one_cube % self.Ngrid                    #   Index of slice 

        #   Extract the correct slice given the index
        embed()


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
    DS = DatasetOfCubes(path, verbose=True)
    DS.__getitem__(5)
    # embed()



