import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from data.h5dataset import DatasetOfCubes
from .COW import COW

generator1 = torch.Generator().manual_seed(42)

class Model:
    def __init__(self, Net, DataPath:str):
        self.Net = Net
        self.DataPath = DataPath
        ###TODO Fix the hard coding below
        self.model = self.Net()
        self.lossFn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def loadData(self, stride=2):
        self.data = DatasetOfCubes(self.DataPath, stride)
        [train, val, test] = random_split(self.data, [0.50, 0.25, 0.25], generator=generator1)
        batch_size = 32
        self.trainLoader = DataLoader(train, batch_size=batch_size)
        self.valLoader = DataLoader(val, batch_size=batch_size)
        self.testLoader = DataLoader(test, batch_size=batch_size)

    def train(self):
        ###TODO Fix the hard coding below
        nrEpochs = 5
        for epoch in range(nrEpochs):
            print(f"Training epoch [{epoch+1}/{nrEpochs}] ...")
            for dataObject in self.trainLoader:
                inputs = dataObject["slice"]
                labels = dataObject["label"]
                yPred = self.model(inputs)
                loss = self.lossFn(yPred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            correct = 0
            count = 0
            tol = 1e-7
            for dataObject in self.valLoader:
                inputs = dataObject["slice"]
                labels = dataObject["label"]
                yPred = self.model(inputs)
                correct += (torch.abs(torch.round(yPred)-labels)<tol).float().sum()
                count += len(labels)
            accuracy = correct / count
            print(f"Epoch [{epoch+1}/{nrEpochs}], Loss: {loss.item():.3f}, accuracy: {accuracy*100:.2f} %")

    def test(self):
        correct = 0
        count = 0
        tol = 1e-7
        for dataObject in self.testLoader:
            inputs = dataObject["slice"]
            labels = dataObject["label"]
            yPred = self.model(inputs)
            correct += (torch.abs(torch.round(yPred) - labels)<tol).float().sum()
            count += len(labels)
        acc = correct / count

        print(f"Correct: [{int(correct)}/{count}] giving accuracy of {acc*100:.2f} %")

if __name__=="__main__":
    print("Model class only")

