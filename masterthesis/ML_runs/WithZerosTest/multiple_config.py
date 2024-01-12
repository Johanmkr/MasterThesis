import numpy as np
import torch.nn as nn

# Params
data_params = {
    "train_test_split": [0.8, 0.2],
    "train_test_seeds": np.arange(0, 75, 1),
    "stride": 1,
    "redshift": 1.0,
    "random_seed": 42,
    "transforms": True,
}

model_params = {
    "input_size": (data_params["stride"], 256, 256),
    "layer_param": 16,
    "activation": nn.LeakyReLU(negative_slope=0.2, inplace=True),
    "output_activation": nn.Identity(),
    "bias": False,
    "dropout": 0.5,
}
cube_frac_in_batch_size = 12
loader_params = {
    "batch_size": int((256 * 3) * cube_frac_in_batch_size),
    "num_workers": 8,
    "prefetch_factor": 2,
}

optimizer_params = {
    "lr": 1e-4,
    "betas": (0.5, 0.999),
    "weight_decay": 1e-11,
}

training_params = {
    "epochs": 1,
    "breakout_loss": 0.0001,
    "tol": 0.5,
}
