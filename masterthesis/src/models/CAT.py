import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchsummary import summary


class CAT(pl.LightningModule):
    def __init__(self, input_channels: int, num_class: int = 2):
        super(CAT, self).__init__()
        self.input_channels = input_channels
        self.num_class = num_class
        self.ngrid = 256

        # LAYER 1 - Convolutional layer with large kernel (input_channels, 256, 256) -> (128, 256, 256)
        self.layer1 = nn.Sequential(
            # (input_channels, 256, 256) -> (128, 256, 256)
            nn.Conv2d(
                self.input_channels, 128, kernel_size=(11, 11), stride=1, padding=10
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                self.ngrid, self.ngrid // 2, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
