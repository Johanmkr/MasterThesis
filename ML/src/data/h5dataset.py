import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from .h5cube import Cube
from .h5collection import Collection

# Used for testing only
# from IPython import embed

class DatasetOfCubes(Dataset):
    def __init__(
            self, 
            data_dir:str,
            stride:int                  = 1,
            Ngrid:int                   = 128,
            boxsize:float               = 5120.0,
            multiple_directories:bool   = False,
            exceptions:str              = None,
            normalise:bool              = True,
            verbose:bool                = False
    ) -> None:
        self.dataDir = data_dir
        self.stride = stride
        self.Ngrid = Ngrid
        self.boxsize = boxsize
        self.multipleDirectories = multiple_directories
        self.exceptions = exceptions
        self.verbose = verbose
        self.normalise = normalise

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
            self.collections.append(Collection(newton_path))
            self.collections.append(Collection(gr_path))

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
        # self.nrSlicesPerColnrDimslection = int(self.totalLength / self.lengthOfCollection)

        self.splitterArray = np.cumsum([self.nrSlicesPerCollection])[:-1]
        self.indicesPerCollection = np.split(np.arange(self.totalLength), self.splitterArray)
        
        # Pre-compute slice indices
        self.slices = []
        for i in range(self.nrSlicesPerDim-1):
            self.slices.append(slice(i*self.stride, (i+1)*self.stride,  1))
        self.slices.append(slice((self.nrSlicesPerDim-1)*self.stride, self.Ngrid, 1))



    def __len__(self):
        return self.totalLength

    def __getitem__(self, long_idx:int) -> dict:
        tol = 1e-7
        temp_idx = np.where(np.abs(np.array(self.indicesPerCollection)-long_idx)<tol)
        collection_idx = int(temp_idx[0])
        cube_idx = int(temp_idx[1]) // self.nrSlicesPerCube
        idx_on_cube = int(temp_idx[1]) % self.nrSlicesPerCube
        axis = idx_on_cube // self.nrSlicesPerDim
        slice_idx = idx_on_cube % self.nrSlicesPerDim

        #   Extract the correct slice given the index
        cube = self.collections[collection_idx][cube_idx]

        if self.verbose:
            RT = f"\n Item information \n"
            RT += f"Collection: {collection_idx}\n"
            RT += f"Cube:       {cube_idx}\n"
            RT += f"Axis:       {axis}\n"
            RT += f"Slice nr.   {slice_idx}\n"
            RT += f"Act. slice: {self.slices[slice_idx]}"
            print(RT)
            print(self.collections[collection_idx])
            print(cube)

        datapoint = {}
        if self.normalise:
            datapoint["slice"] = torch.tensor(self._normalise_array(self._slice_cube_along_axis(cube, axis, slice_idx)), dtype=torch.float32)
        else:
            datapoint["slice"] = torch.tensor(self._slice_cube_along_axis(cube, axis, slice_idx), dtype=torch.float32)
        datapoint["label"] = torch.tensor([cube.gr], dtype=torch.float32)

        # To test network
        # if cube.gr:
        #     datapoint["slice"] = datapoint["slice"] * 5
        return datapoint


    def _find_directories(self) -> None:
        directory_entries = os.listdir(self.dataDir)
        if self.exceptions is not None:
            for entry in directory_entries not in self.exceptions:
                self.directories.append(entry)
            self.directories.sort()
        else:
            self.directories = directory_entries.sort()

    def _slice_cube_along_axis(self, cube:Cube, axis:int, slice_idx:int) -> np.ndarray:
        local_slices = [slice(None)] * self.nrDims 
        local_slices[axis] = self.slices[slice_idx]
        return cube[tuple(local_slices)].reshape(self.stride, self.Ngrid,self.Ngrid)
    
    def _normalise_array(self, arr:np.ndarray) -> np.ndarray:
        return (arr-arr.mean())/arr.std()

        
if __name__=="__main__":
    output_path = os.path.abspath("").replace("Summer-Sandbox23/ML/src/data", "NbodySimulation/gevolution-1.2/output")
    run_type = "/intermediate"

    path = output_path + run_type
    DS = DatasetOfCubes(path, stride=2, verbose=False)
    print(DS[0])
    # inp = True
    # while inp:
    #     num = int(input("Write index: "))
    #     DS[num]
        # inp = False if input("Continue [y/n]?") in ["n", "no", "N"] else True
    



