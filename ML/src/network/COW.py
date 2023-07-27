import numpy as np
import torch
from torch import nn
from torchsummary import summary
# from .network import Network

class COW(nn.Module):
    def __init__(self, inputSize:tuple=(2,128,128)):
        super(COW, self).__init__()
        self.inputSize = inputSize
        self.depth = inputSize[0]
        self.width = inputSize[1]
        self.height = inputSize[2]

        ### LAYER 1 (Convolutional) ### (1, 64, 64) -> (64, 64, 64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.depth, 128, kernel_size=(3,3), stride=1, padding=1),       # (1, 64, 64) -> (64, 64, 64)
            nn.ReLU(),                                                      # -
            nn.Dropout(0.25),                                                # -
        )
        
        ### LAYER 2 (Convolutional) ### (64, 64, 64) -> (32, 16, 16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(3,3), stride=1, padding=1),      # (64, 64, 64) -> (32, 64, 64)
            nn.ReLU(),                                                      # - 
            nn.MaxPool2d(kernel_size=(8,8)),                                # (32, 64, 64) -> (32, 16, 16)
        )

        ### LAYER 3 (Fully connected) ###   (8192) -> (256)
        self.layer3 = nn.Sequential(
            nn.Flatten(),                                                   # (32, 16, 16) -> (8192)
            nn.Linear(int(32*16*16), 256),                                  # (8192) -> (256)
            nn.ReLU(),                                                      # -
            nn.Dropout(0.25),                                               # -
        )

        ### LAYER 4 (Fully connected) ###
        self.layer4 = nn.Sequential(
            nn.Linear(256, 16),                                             # (256) -> (16)
            nn.ReLU(),                                                      # -
            nn.Dropout(0.25),                                               # -
        )

        ### LAYER 5 (Output) ###
        self.output = nn.Sequential(
            nn.Linear(16, 1),                                               # (16) -> (2)
            nn.Sigmoid(),
        )

        # List of layers
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return self.output(X)
    
    def printSummary(self, input:tuple=(2,128,128)):
        summary(self, self.inputSize)


if __name__=="__main__":
    print("Network class only")