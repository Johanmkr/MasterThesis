"""
    3D convolutional neural network for classifying whole datacubes from GR/Newtonian simulations.
"""

import torch
import torch.nn as nn


class SLOTH(nn.Module):
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

        # LAYER - 3D convolutional layer (num_channels, 256, 256, 256) -> (4*layer_param, 64, 64, 64)
        self.conv3d1 = nn.Sequential(
            nn.Conv3d(
                1,
                layer_param * 4,
                kernel_size=(4, 4, 4),
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.BatchNorm3d(layer_param * 4),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # LAYER - 3D convolutional layer (4*layer_param, 64, 64, 64) -> (5*layer_param, 16, 32, 32)
        self.conv3d2 = nn.Sequential(
            nn.Conv3d(
                layer_param * 4,
                layer_param * 5,
                kernel_size=(4, 3, 3),
                stride=(2, 1, 1),
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.BatchNorm3d(layer_param * 5),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # LAYER - 3D convolutional layer (5*layer_param, 16, 32, 32) -> (1, 8, 16, 16)
        self.conv3d3 = nn.Sequential(
            nn.Conv3d(
                layer_param * 5,
                1,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.BatchNorm3d(1),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - 2D convolutional layer (8, 16, 16) -> (4*layer_param, 16, 16)
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(
                8,
                layer_param * 4,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=self.bias,
            ),
            nn.BatchNorm2d(layer_param * 4),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - 2D convolutional layer (4*layer_param, 16, 16) -> (8*layer_param, 8, 8)
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(
                layer_param * 4,
                layer_param * 8,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 8),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - 2D convolutional layer (8*layer_param, 8, 8) -> (16*layer_param, 1, 1)
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(
                layer_param * 8,
                layer_param * 16,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=1,
                bias=self.bias,
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(layer_param * 16),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - fully connected layer (16*layer_param) -> (8*layer_param)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layer_param * 16 * 2 * 2, layer_param * 8),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - fully connected layer (8*layer_param) -> (layer_param)
        self.fc2 = nn.Sequential(
            nn.Linear(layer_param * 8, layer_param),
            self.activation,
            nn.Dropout(self.dropout),
        )

        # Layer - output layer (layer_param) -> (1)
        self.output = nn.Sequential(
            nn.Linear(layer_param, 1),
            self.output_activation,
        )

    def forward(self, X):
        X = torch.unsqueeze(X, dim=1)
        X = self.conv3d1(X)
        X = self.conv3d2(X)
        X = self.conv3d3(X)

        # Reshape for 2D convolutional layers
        X = torch.reshape(X, (-1, 8, 16, 16))
        
        X = self.conv2d1(X)
        X = self.conv2d2(X)
        X = self.conv2d3(X)
        
        X = self.fc1(X)
        X = self.fc2(X)
        out = self.output(X)
        return out


if __name__ == "__main__":
    print("SLOTH model for predicting whole datacubes from GR/Newtonian simulations.")
