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

### Initialisation
Actually running a session/training required an initiation file. This is a template of such file, let's call it `temp.yaml`:
 
 ```yaml
 model_params:
  architecture: "RACCOON"         # "RACCOON" or "PENGUIN": Network architecture.
  layer_param: 16                 # (int) Parameter to control the number of feature maps/complexity of the network. 
  dropout: 0.5                    # (float) Dropout rate of fully connected layers. Must be a probability between 0 and 1.
  name: "Somename"                # (str) Name of model/ NB: In order to continue a run, the name (and other setting must be identical).

data_params:
  datafile: "/mn/stornext/d10/data/johanmkr/simulations/data_z1/data_z1.h5"
  train_seeds: 150                # (int) Number of seeds used for training [0, train_seeds).
  test_seeds: 100                 # (int) Number of seeds used for testing: [train_seeds, train_seeds+test_seeds).
  newton_augmentation: 1.0        # (float) Factor to change the newtonian images. Should be != 1 for test purposes only. 

train_params:
  epochs: 10                      # (int) Nr. epoch to train for (can be trained before as well, will just continue).
  lr: 0.001                       # (float) Learning rate for optimizer.
  log_dir: "testruns"             # (str) Directory to log the training. This can be read by tensorboard later.
```
More parameters can be changed in the file [`run_controller.py`](run_controller.py) (for instance multi- or single-gpu training). This file is also used to run the training scripts, with the initialization file as a command line argument:

    python3 run_controller.py temp.yaml

This automatically trains on multiple GPUs if they are available, and also searches for a model with the same name as the one in the initialization in order to continue training. 