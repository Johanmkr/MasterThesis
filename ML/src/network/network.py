from src.MLutils import *

class Network(nn.Module):
    def __init__(self, inputSize:tuple):
        super(Network, self).__init__()
        self.inputSize = inputSize
        self.nrChannels = inputSize[0]
        self.width = inputSize[1]
        self.height = inputSize[2]

    def forward(self, X):
        pass

if __name__=="__main__":
    print("Network class only")