import numpy as np
import configparser
import os

class Handler:
    def __init__(self, h5Path:str) -> None:
        """Constructor

        Args:
            h5Path (str): Path to directory. 
        """
        self.h5Path = h5Path

    def _Extracth5Data(self) -> list:
        """Extracts the data from all .h5 files in the directory provided to the constructor

        Returns:
            list[np.ndarray]: Data from each file, as element in list
        """
        ###TODO Update this, make us of class h5Cube
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
    pass