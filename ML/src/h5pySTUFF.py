from MLutils import *

class h5Handler:
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
    

class h5Cube:
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
    DataPath = os.path.abspath("").replace("Summer-Sandbox23/ML/src", "NbodySimulation/gevolution-1.2/output/intermediate/")
    # obj = h5Handler(DataPath)
    # data = obj()
    testobj = h5Cube(DataPath + "gr/lcdm_snap000_phi.h5")
    # testobj._readCube()
    data = testobj()
    print(testobj)
