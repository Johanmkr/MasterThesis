import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchsummary import summary
import numpy as np


class CAT(pl.LightningModule):
    def __init__(self, input_channels: int, num_class: int = 2):
        super(CAT, self).__init__()
        self.input_channels = input_channels
        self.num_class = num_class
        self.ngrid = 256

        # LAYER 1 - Convolutional layer with large kernel
        layer_1_args = {
            "kernel_size": 11,
            "stride": 1,
            "padding": 5,
        }
        layer1_out_shape = self._get_side_shape_out(self.ngrid, **layer_1_args)
        self.layer1 = nn.Sequential(
            # (input_channels, 256, 256) -> (32, 256, 256)
            nn.Conv2d(self.input_channels, 32, **layer_1_args),
            nn.ReLU(),
            nn.Dropout(0.10),
        )

        # LAYER 2 - Convolutional layer with dilated kernel
        layer2_args = {
            "kernel_size": 9,
            "stride": 1,
            "padding": 5,
            "dilation": 5,
        }
        layer2_out_shape = self._get_side_shape_out(layer1_out_shape, **layer2_args)
        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_out_shape, 32, **layer2_args),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

    def _get_side_shape_out(
        self,
        VAL_in: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> int:
        return np.floor(
            (VAL_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
