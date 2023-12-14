import torch
import torch.nn as nn
from torchsummary import summary


class MOTH(nn.Module):
    def __init__(self, input_size: tuple, layer_param: float = 4):
        super().__init__()
        self.layers = []
        self.input_size = input_size
        try:
            num_channels = input_size[0]
            depth = input_size[1]
            width = input_size[2]
            height = input_size[3]
        except IndexError:
            depth = input_size[0]
            width = input_size[1]
            height = input_size[2]

        # LAYER - convolutional layer (depth, 256, 256) -> (layer_param, 64, 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(depth, layer_param, kernel_size=(4, 4), stride=2, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.conv1)

        # Layer - convolutional layer (layer_param, 64, 64) -> (2*layer_param, 16, 16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                layer_param, layer_param * 2, kernel_size=(4, 4), stride=2, padding=1
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 2),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.conv2)

        # Layer - convolutional layer (2*layer_param, 16, 16) -> (8*layer_param, 4, 4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                layer_param * 2,
                layer_param * 8,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 8),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.conv3)

        # Layer - convolutional layer (8*layer_param, 4, 4) -> (16*layer_param, 1, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                layer_param * 8,
                layer_param * 16,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 16),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.conv4)

        # Layer - fully connected layer (16*layer_param) -> (8*layer_param)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layer_param * 16, layer_param * 8),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.fc1)

        # Layer - fully connected layer (8*layer_param) -> (layer_param)
        self.fc2 = nn.Sequential(
            nn.Linear(layer_param * 8, layer_param),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.fc2)

        # Layer - output layer (layer_param) -> (1)
        self.output = nn.Sequential(
            nn.Linear(layer_param, 1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return self.output(X)

    def printSummary(self):
        summary(self, self.input_size)


if __name__ == "__main__":
    print("MOTH network class only!")
