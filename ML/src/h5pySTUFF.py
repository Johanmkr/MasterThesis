from MLutils import *

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
        initFile = [self.h5Path + name for name in allFiles if ".ini" in name][0]
        config = configparser.ConfigParser()
        config.read_string('[default]\n' + open(initFile).read())
        self.configDict = {}
        for option in config.options('default'):
            value = config.get('default', option)
            self.configDict[option] = value
        h5Files = [self.h5Path + name for name in allFiles if ".h5" in name]
        h5Datasets = {}
        for i, filename in enumerate(h5Files):
            name = self.configDict["snapshot outputs"].split(",").strip()[i]
            h5Datasets[name] = self._Extracth5Specifics(filename)
        return h5Datasets
    
    def __call__(self):
        return self._Extracth5Data()
    
    
if __name__=="__main__":
    DataPath = os.path.abspath("").replace("Summer-Sandbox23/ML/src", "NbodySimulation/gevolution-1.2/output/test_intermediate/")
    obj = h5Handler(DataPath)
    data = obj()
