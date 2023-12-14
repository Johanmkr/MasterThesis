"""
    Smaller 2D convolutional neural network for classification of slices of datacubes from GR/Newtonian simulations.
"""


import torch
import torch.nn as nn


class RACOON(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        layer_param: float = 64,
        activation=nn.ReLU(),
        output_activation=nn.Sigmoid(),
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
        
        