"""
    2D convolutional neural network for classification of slices of datacubes from GR/Newtonian simulations.
"""

import torch
import torch.nn as nn


class PENGUIN(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        layer_param: float = 4,
        activation=nn.LeakyReLU(),
        output_activation=nn.Sigmoid(),
        bias=False,
        dropout=0.25,
    ):
        super().__init__()
        self.input_size = input_size
        self.activation = activation
        self.output_activation = output_activation
        self.bias = bias
        self.dropout = dropout
        self.num_channels = self.input_size[0]

        # LAYER - 2D convolutional layer (num_channels, 256, 256) -> (4*layer_param, 128, 128)
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(
                self.num_channels,
                layer_param * 4,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 4),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # LAYER - 2D convolutional layer (4*layer_param, 128, 128) -> (6*layer_param, 64, 64)
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(
                layer_param * 4,
                layer_param * 6,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 6),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # LAYER - 2D convolutional layer (6*layer_param, 64, 64) -> (8*layer_param, 32, 32)
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(
                layer_param * 6,
                layer_param * 8,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 8),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - 2D convolutional layer (8*layer_param, 32, 32) -> (10*layer_param, 16, 16)
        self.conv2d4 = nn.Sequential(
            nn.Conv2d(
                layer_param * 8,
                layer_param * 10,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 10),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - Fully connected layer (10*layer_param, 16, 16) -> (10*layer_param)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layer_param * 10 * 16 * 16, layer_param * 10),
            nn.BatchNorm1d(layer_param * 10),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - Fully connected layer (10*layer_param) -> (layer_param)
        self.fc2 = nn.Sequential(
            nn.Linear(layer_param * 10, layer_param),
            nn.BatchNorm1d(layer_param),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer -output layer (layer_param) -> (1)
        self.output = nn.Sequential(
            nn.Linear(layer_param, 1),
            self.output_activation,
        )

    def forward(self, X):
        for layer in [
            self.conv2d1,
            self.conv2d2,
            self.conv2d3,
            self.conv2d4,
            self.fc1,
            self.fc2,
        ]:
            X = layer(X)
        return self.output(X)
