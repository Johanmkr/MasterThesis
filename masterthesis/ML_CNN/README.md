# Training Convolutional Neural Networks. 

## Architecture

The file [`architectures.py`](architectures.py) contain the two PyTorch network architecture. Their structure is outlined as:

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

where `layer_param` is an integer that control the size/complexity of the network. `num_channels` is the number of input channels and will be controlled by the first dimension of the input images (usually 1). 

Each layer consist of:
 * A convolutional operation (mathematically a cross-correlation). The choice of kernel size, stride, and padding takes care of the downsizing of the images, effectively making a pooling layer redundant. 
 * Batch normalization, of the same size as the output image. 
 * Activation function (normal ReLU by default). 


## Data

The file [`data.py`](data.py) has routines for creating the dataset. The two main routines are:
* `WholeCubeDataset` -> Load dataset where datacubes have been pre-processed and stored in a single `.h5`-file. These cubes are already standardized to the mean and variance of the whole dataset (2000 seeds), for the specific redshift. Performs lazy loading.
* `WholeCubeDataset2` -> Routine used for generating the first dataset of the amplified data (large $A_s$). Meant to be used for smaller dataset, loads all cubes into CPU-memory and then standardize the set based on the mean and variance of the loaded cubes. 

There is also a routing that creates testing and training datasets. Which of the abovementioned routines to use for this must as of now be manually changed. 

## Making a run for it

## Trainers

### Single-GPU

### Multi-GPU

## Testing