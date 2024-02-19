import torch
import torch.nn as nn

# Write a string summary of RACCON and PENGUIN architectures:

""" 
RACOON:
    - 2D convolutional layer (num_channels, 256, 256) -> (layer_param, 64, 64)
    - 2D convolutional layer (layer_param, 64, 64) -> (4*layer_param, 16, 16)
    - 2D convolutional layer (4*layer_param, 16, 16) -> (8*layer_param, 4, 4)
    - Fully connected layer (8*layer_param * 4 * 4) -> (layer_param)
    - Output layer (layer_param) -> (1)

PENGUIN:
    - 2D convolutional layer (num_channels, 256, 256) -> (layer_param, 128, 128)
    - 2D convolutional layer (layer_param, 128, 128) -> (2*layer_param, 64, 64)
    - 2D convolutional layer (2*layer_param, 64, 64) -> (4*layer_param, 32, 32)
    - 2D convolutional layer (4*layer_param, 32, 32) -> (6*layer_param, 16, 16)
    - 2D convolutional layer (6*layer_param, 16, 16) -> (8*layer_param, 8, 8)
    - 2D convolutional layer (8*layer_param, 8, 8) -> (10*layer_param, 4, 4)
    - Fully connected layer (10*layer_param * 4 * 4) -> (10*layer_param)
    - Fully connected layer (10*layer_param) -> (layer_param)
    - Output layer (layer_param) -> (1)
"""


class RACOON(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        layer_param: float = 64,
        activation=nn.ReLU(),
        output_activation=nn.Identity(),
        dropout=0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.num_channels = self.input_size[0]
        self.conv_layers = []
        self.fc_layers = []
        self.convBias = False

        # LAYER - 2D convolutional layer (num_channels, 256, 256) -> (layer_param, 64, 64)
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(
                self.num_channels,
                layer_param,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=self.convBias,
            ),
            nn.BatchNorm2d(layer_param),
            self.activation,
        )
        self.conv_layers.append(self.conv2d1)

        # LAYER - 2D convolutional layer (layer_param, 64, 64) -> (4*layer_param, 16, 16)
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(
                layer_param,
                layer_param * 4,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=self.convBias,
            ),
            nn.BatchNorm2d(layer_param * 4),
            self.activation,
        )
        self.conv_layers.append(self.conv2d2)

        # LAYER - 2D convolutional layer (4*layer_param, 16, 16) -> (8*layer_param, 4, 4)
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(
                layer_param * 4,
                layer_param * 8,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=self.convBias,
            ),
            nn.BatchNorm2d(layer_param * 8),
            self.activation,
        )
        self.conv_layers.append(self.conv2d3)

        # LAYER - Fully connected layer (8*layer_param * 4 * 4) -> (layer_param)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layer_param * 8 * 4 * 4, layer_param),
            self.activation,
            nn.Dropout(self.dropout),
        )
        self.fc_layers.append(self.fc1)

        # LAYER - Output layer (layer_param) -> (1)
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


class PENGUIN(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        layer_param: float = 64,
        activation=nn.ReLU(),
        output_activation=nn.Identity(),
        dropout=0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.num_channels = self.input_size[0]
        self.conv_layers = []
        self.fc_layers = []
        self.convBias = False

        # LAYER - 2D convolutional layer (num_channels, 256, 256) -> (layer_param, 128, 128)
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(
                self.num_channels,
                layer_param,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.convBias,
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
                bias=self.convBias,
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
                bias=self.convBias,
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
                bias=self.convBias,
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
                bias=self.convBias,
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
                bias=self.convBias,
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


class model_o3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_o3_err, self).__init__()

        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C01 = nn.Conv2d(
            channels,
            2 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C02 = nn.Conv2d(
            2 * hidden,
            2 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C03 = nn.Conv2d(
            2 * hidden,
            2 * hidden,
            kernel_size=2,
            stride=2,
            padding=0,
            padding_mode="circular",
            bias=True,
        )
        self.B01 = nn.BatchNorm2d(2 * hidden)
        self.B02 = nn.BatchNorm2d(2 * hidden)
        self.B03 = nn.BatchNorm2d(2 * hidden)

        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(
            2 * hidden,
            4 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C12 = nn.Conv2d(
            4 * hidden,
            4 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C13 = nn.Conv2d(
            4 * hidden,
            4 * hidden,
            kernel_size=2,
            stride=2,
            padding=0,
            padding_mode="circular",
            bias=True,
        )
        self.B11 = nn.BatchNorm2d(4 * hidden)
        self.B12 = nn.BatchNorm2d(4 * hidden)
        self.B13 = nn.BatchNorm2d(4 * hidden)

        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(
            4 * hidden,
            8 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C22 = nn.Conv2d(
            8 * hidden,
            8 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C23 = nn.Conv2d(
            8 * hidden,
            8 * hidden,
            kernel_size=2,
            stride=2,
            padding=0,
            padding_mode="circular",
            bias=True,
        )
        self.B21 = nn.BatchNorm2d(8 * hidden)
        self.B22 = nn.BatchNorm2d(8 * hidden)
        self.B23 = nn.BatchNorm2d(8 * hidden)

        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(
            8 * hidden,
            16 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C32 = nn.Conv2d(
            16 * hidden,
            16 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C33 = nn.Conv2d(
            16 * hidden,
            16 * hidden,
            kernel_size=2,
            stride=2,
            padding=0,
            padding_mode="circular",
            bias=True,
        )
        self.B31 = nn.BatchNorm2d(16 * hidden)
        self.B32 = nn.BatchNorm2d(16 * hidden)
        self.B33 = nn.BatchNorm2d(16 * hidden)

        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(
            16 * hidden,
            32 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C42 = nn.Conv2d(
            32 * hidden,
            32 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C43 = nn.Conv2d(
            32 * hidden,
            32 * hidden,
            kernel_size=2,
            stride=2,
            padding=0,
            padding_mode="circular",
            bias=True,
        )
        self.B41 = nn.BatchNorm2d(32 * hidden)
        self.B42 = nn.BatchNorm2d(32 * hidden)
        self.B43 = nn.BatchNorm2d(32 * hidden)

        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(
            32 * hidden,
            64 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C52 = nn.Conv2d(
            64 * hidden,
            64 * hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.C53 = nn.Conv2d(
            64 * hidden,
            64 * hidden,
            kernel_size=2,
            stride=2,
            padding=0,
            padding_mode="circular",
            bias=True,
        )
        self.B51 = nn.BatchNorm2d(64 * hidden)
        self.B52 = nn.BatchNorm2d(64 * hidden)
        self.B53 = nn.BatchNorm2d(64 * hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(
            64 * hidden,
            128 * hidden,
            kernel_size=4,
            stride=1,
            padding=0,
            padding_mode="circular",
            bias=True,
        )
        self.B61 = nn.BatchNorm2d(128 * hidden)

        self.P0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1 = nn.Linear(128 * hidden, 64 * hidden)
        self.FC2 = nn.Linear(64 * hidden, 1)

        self.dropout = nn.Dropout(p=dr)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.Linear)
            ):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, image):

        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))

        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))

        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))

        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))

        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0], -1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        # y[:,6:12] = torch.square(x[:,6:12])

        return y
