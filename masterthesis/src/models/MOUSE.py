from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
from torchsummary import summary
import torchvision
import numpy as np

from IPython import embed


class MOUSE(pl.LightningModule):
    def __init__(self, input_channels, ngrid, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.input_channels = input_channels
        self.ngrid = ngrid
        self.layers = []
        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy()

        # LAYER - fully connected layer
        self.fc1 = nn.Sequential(
            nn.flatten(),
            nn.Linear(self.input_channels * self.ngrid * self.ngrid, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.fc1)

        # LAYER - fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers.append(self.fc2)

        # LAYER - output layer
        self.output = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layers.append(self.output)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _common_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.accuracy(y_hat, y),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        X, y = batch
        y_hat = self.forward(X)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
