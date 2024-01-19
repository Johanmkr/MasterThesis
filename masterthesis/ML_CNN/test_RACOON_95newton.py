VERBOSE = True
MULTIPLE_GPUS = True

import numpy as np
import torch.nn as nn

import architectures as arch
import train_multigpu as train

# Params
data_params = {
    "train_test_split": [0.8, 0.2],
    "train_test_seeds": np.arange(0, 1750, 1),
    "stride": 1,
    "redshift": 1.0,
    "random_seed": 42,
    "transforms": True,
    "newton_augmentation": 0.95,  # must be 1.0 for serious training, change for testing pipeline.
}

architecture_params = {
    "input_size": (data_params["stride"], 256, 256),
    "layer_param": 64,
    "activation": nn.ReLU(),
    "output_activation": nn.Identity(),
    "bias": False,
    "dropout": 0.5,
}
model_name = f"RACOON_test_95newton_z{data_params['redshift']:.0f}_lp{architecture_params['layer_param']}"
model_params = {
    "architecture": arch.RACOON,
    "model_name": model_name,
    "load_model": True,
    "model_save_path": f"models/{model_name}.pt",
}

cube_frac_in_batch_size = 3
loader_params = {
    "batch_size": int((256 * 3) * cube_frac_in_batch_size),
    "num_workers": 0,
    "prefetch_factor": None,
    "pin_memory": False,
    "shuffle": True,
    "drop_last": True,
}

optimizer_params = {
    "lr": 1e-2,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-8,
}

training_params = {
    "epochs": 1,
    "breakout_loss": 1e-2,
    "tol": 0.25,
    "writer_log_path": f"testruns/{model_params['model_name']}_{optimizer_params['lr']:.5f}",
    "test_every": 2,
}


if __name__ == "__main__":
    print(f"MULTIPLE_GPUS: {MULTIPLE_GPUS}")
    # print all info from config file
    if VERBOSE:
        print("CONFIGURATION:")
        for dicti in [
            data_params,
            model_params,
            loader_params,
            optimizer_params,
            training_params,
        ]:
            for key, value in dicti.items():
                print(f"{key}: {value}")
            print()
    train.train(
        data_params,
        architecture_params,
        model_params,
        loader_params,
        optimizer_params,
        training_params,
    )
