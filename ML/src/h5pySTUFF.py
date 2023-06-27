from src.MLutils import *

class h5Handler:
    def __init__(self, h5Path:str) -> None:
        """Constructor

        Args:
            h5Path (str): Path to directory. 
        """
        self.h5Path = h5Path

    def _Extracth5Specifics(self, filename:str) -> np.ndarray:
        """Extract the data as numpy array given a specific .h5 file

        Args:
            filename (str): Name of .h5 file

        Returns:
            np.ndarray: Data of file.
        """
        h5File = h5py.File(filename, "r")
        h5Dataset = h5File["data"][()]
        h5File.close()
        return h5Dataset

    def _Extracth5Data(self) -> list:
        """Extracts the data from all .h5 files in the directory provided to the constructor

        Returns:
            list[np.ndarray]: Data from each file, as element in list
        """
        allFiles = os.listdir(self.h5Path)
        h5Files = [self.h5Path + name for name in allFiles if ".h5" in name]
        h5Datasets = []
        for filename in h5Files:
            h5Datasets.append(self._Extracth5Specifics(filename))
        return h5Datasets
    
    def __call__(self):
        return self._Extracth5Data()
    
