"""
    2D convolutional neural network for classification of slices of datacubes from GR/Newtonian simulations.
"""

import torch
import torch.nn as nn


class PENGUIN(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        layer_param: float = 64,
        activation=nn.ReLU(),
        output_activation=nn.Identity(),
        bias=False,
        dropout=0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.activation = activation
        self.output_activation = output_activation
        self.bias = bias
        self.dropout = dropout
        self.num_channels = self.input_size[0]
        self.conv_layers = []
        self.fc_layers = []

        # LAYER - 2D convolutional layer (num_channels, 256, 256) -> (layer_param, 128, 128)
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(
                self.num_channels,
                layer_param,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.BatchNorm2d(layer_param),
            self.activation,
        )
        self.conv_layers.append(self.conv2d1)

        # LAYER - 2D convolutional layer (layer_param, 128, 128) -> (2*layer_param, 64, 64)
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(
                layer_param,
                layer_param * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.BatchNorm2d(layer_param * 2),
            self.activation,
        )
        self.conv_layers.append(self.conv2d2)

        # LAYER - 2D convolutional layer (2*layer_param, 64, 64) -> (4*layer_param, 32, 32)
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(
                layer_param * 2,
                layer_param * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.BatchNorm2d(layer_param * 4),
            self.activation,
        )
        self.conv_layers.append(self.conv2d3)

        # LAYER - 2D convolutional layer (4*layer_param, 32, 32) -> (6*layer_param, 16, 16)
        self.conv2d4 = nn.Sequential(
            nn.Conv2d(
                layer_param * 4,
                layer_param * 6,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.BatchNorm2d(layer_param * 6),
            self.activation,
        )
        self.conv_layers.append(self.conv2d4)

        # LAYER - 2D convolutional layer (6*layer_param, 16, 16) -> (8*layer_param, 8, 8)
        self.conv2d5 = nn.Sequential(
            nn.Conv2d(
                layer_param * 6,
                layer_param * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.BatchNorm2d(layer_param * 8),
            self.activation,
        )
        self.conv_layers.append(self.conv2d5)

        # LAYER - 2D convolutional layer (8*layer_param, 8, 8) -> (10*layer_param, 4, 4)
        self.conv2d6 = nn.Sequential(
            nn.Conv2d(
                layer_param * 8,
                layer_param * 10,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.BatchNorm2d(layer_param * 10),
            self.activation,
        )
        self.conv_layers.append(self.conv2d6)

        # LAYER - Fully connected layer (10*layer_param, 4, 4) -> (10*layer_param)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layer_param * 10 * 4 * 4, layer_param * 10),
            nn.BatchNorm1d(layer_param * 10),
            self.activation,
            nn.Dropout(self.dropout),
        )
        self.fc_layers.append(self.fc1)

        # LAYER - Fully connected layer (10*layer_param) -> (layer_param)
        self.fc2 = nn.Sequential(
            nn.Linear(layer_param * 10, layer_param),
            nn.BatchNorm1d(layer_param),
            self.activation,
            nn.Dropout(self.dropout),
        )
        self.fc_layers.append(self.fc2)

        # LAYER -output layer (layer_param) -> (1)
        self.output = nn.Sequential(
            nn.Linear(layer_param, 1),
            self.output_activation,
        )
        self.fc_layers.append(self.output)

    def forward(self, X):
        for layer in self.conv_layers:
            X = layer(X)
        for layer in self.fc_layers:
            X = layer(X)
        return X
