import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
from torchsummary import summary
import numpy as np

from IPython import embed


class CAT(pl.LightningModule):
    def __init__(self, input_channels: int, num_class: int = 2):
        super(CAT, self).__init__()
        self.input_channels = input_channels
        self.num_class = num_class
        self.ngrid = 256
        self.layers = []

        # LAYER - Convolutional layer with large kernel
        conv1_args = {
            "kernel_size": 11,
            "stride": 1,
            "padding": 5,
        }
        # embed()
        conv1_out_shape = self._get_side_shape_out(self.ngrid, **conv1_args)
        self.conv1 = nn.Sequential(
            # (input_channels, 256, 256) -> (32, 256, 256)
            nn.Conv2d(self.input_channels, 32, **conv1_args),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.layers.append(self.conv1)

        # LAYER - Convolutional layer with dilated kernel
        conv2_args = {
            "kernel_size": 9,
            "stride": 1,
            "padding": 5,
            "dilation": 5,
        }
        conv2_out_shape = self._get_side_shape_out(conv1_out_shape, **conv2_args)
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_out_shape, 32, **conv2_args),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.conv2)

        # LAYER - Convolutional layer with maxpool ans small kernel
        conv3_args = {
            "kernel_size": 5,
            "stride": 1,
            "padding": 1,
        }
        maxpool_args = {
            "kernel_size": 3,
            "stride": 3,
            "padding": 0,
            "dilation": 1,
        }
        conv3_out_shape = self._get_side_shape_out(conv2_out_shape, **conv3_args)
        maxpool_out_shape = self._get_side_shape_out(conv3_out_shape, **maxpool_args)
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv2_out_shape, 32, **conv3_args),
            nn.ReLU(),
            nn.MaxPool2d(**maxpool_args),
        )
        self.layers.append(self.conv3)

        # LAYER - First fully connected layer
        fc1_args = {
            "in_features": maxpool_out_shape**2 * 32,
            "out_features": 256,
        }
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(**fc1_args),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.fc1)

        # LAYER - Second fully connected layer
        fc2_args = {
            "in_features": 256,
            "out_features": 16,
        }
        self.fc2 = nn.Sequential(
            nn.Linear(**fc2_args),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.fc2)

        # LAYER - Output layer
        fc3_args = {
            "in_features": 16,
            "out_features": self.num_class,
        }
        self.fc3 = nn.Sequential(
            nn.Linear(**fc3_args),
            nn.Sigmoid(),
        )
        self.layers.append(self.fc3)

        # Others
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = BinaryAccuracy()

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X)
        return X

    def printSummary(self) -> None:
        input = (self.input_channels, self.ngrid, self.ngrid)
        summary(self, input)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        X, y = batch
        y_hat = self.forward(X)
        loss = loss.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(y_hat, y))
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        X, y = batch
        y_hat = self.forward(X)
        loss = loss.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(y_hat, y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _get_side_shape_out(
        self,
        VAL_in: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> int:
        return int(
            np.floor(
                np.array(
                    [
                        (VAL_in + 2 * padding - dilation * (kernel_size - 1) - 1)
                        / stride
                        + 1
                    ]
                ),
            )
        )
