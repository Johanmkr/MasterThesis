from src.MLutils import *
###TODO modify this to take a list of cubes as input
class TestCubes(Dataset):
    def __init__(self, newtonCube, grCube, stride=1, transform=None, additionalInfo=False):
        self.newtonCube = newtonCube
        self.grCube = grCube
        self.stride=stride
        self.length = self.__len__()
        self.halflength = int(self.length/2.)
        self.transform = transform
        self.additionalInfo = additionalInfo

    def __len__(self):
        newtonShape = self.newtonCube.shape
        grShape = self.grCube.shape
        newtonLength = 0
        grLength = 0
        for i in range(len(newtonShape)):
            newtonLength += newtonShape[i]
            grLength += grShape[i]
        return int((newtonLength+grLength)/self.stride)

    def _getSlice(self, data:np.ndarray, axis:int, index:int or list or tuple):
        slices = [slice(None)] * data.ndim
        if isinstance(index, int):
            slices[axis] = index
        else:
            slices[axis] = slice(index[0], index[1])
        return data[tuple(slices)]

    def __getitem__(self, idx):
        NEWTON = idx < self.halflength
        if not NEWTON:
            idx = idx-self.halflength
        axis = idx // self.newtonCube.shape[0] #only works for cubic cubes and stride 1 for now, should be easy to improve
        index = idx % self.newtonCube.shape[0]
        if NEWTON:
            slice_data = self._getSlice(self.newtonCube, axis, index)
            label = torch.tensor([0.0], dtype=torch.float32)
        else:
            slice_data = self._getSlice(self.grCube, axis, index)
            label = torch.tensor([1.0], dtype=torch.float32)
        if self.stride != 1:
            sample = {"image": torch.tensor(slice_data, dtype=torch.float32), "label": label}
        else:
            sample = {"image": torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0), "label": label}

        if self.additionalInfo:
            sample["axis"] = axis
            sample["index"] = index
        if self.transform:
            ###TODO Need to fix these issuse with transformation, normalisation for now
            # toBeNormalized = sample["image"]
            # Normalized = self.transform(toBeNormalized)
            # sample["image"] = Normalized
            sample["image"] = (sample["image"]-torch.mean(sample["image"]))/torch.std(sample["image"])

        return sample
    
    def __str__(self, idx=None):
        returnString = "Dataset info:\n----------------------\n"
        returnString += f"  Newton cube size: {self.newtonCube.shape}\n"
        returnString += f"  GR cube size: {self.grCube.shape}\n"
        returnString += f"  Stride: {self.stride}\n"
        return returnString

    def printImage(self, idx):
        returnString = ""
        sample = self.__getitem__(idx)
        image = sample["image"]
        returnString += f"Image info (Newton:0, GR:1):\n"
        for key, val in sample.items():
            if key != "image":
                returnString += f"  {key}: {val}\n"
        returnString += "Basic statistics:\n"
        basicStat = {
            "mean": torch.mean(image),
            "min": torch.min(image),
            "max": torch.max(image),
            "std": torch.std(image),
            "median": torch.median(image)
        }
        for key, val in basicStat.items():
            returnString += f"  {key}: {val}\n" 
        returnString += "\n"
        return returnString
    
if __name__=="__main__":
    print("Data handler class only")