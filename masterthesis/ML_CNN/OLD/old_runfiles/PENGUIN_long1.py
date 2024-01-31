VERBOSE = True
MULTIPLE_GPUS = True

import numpy as np
import torch.nn as nn

import architectures as arch

if MULTIPLE_GPUS:
    import train_multigpu as train
else:
    import train_singlegpu as train

# Params
train_test_division = 1500
total_seeds = 1700
data_params = {
    "train_seeds": np.arange(0, train_test_division, 1),
    "test_seeds": np.arange(train_test_division, total_seeds, 1),
    "redshift": 1.0,
    "newton_augmentation": 1.0,  # must be one for serious training, change for testing pipeline.
    "target_noise": 0.0,
}

architecture_params = {
    "input_size": (1, 256, 256),
    "layer_param": 32,
    "activation": nn.LeakyReLU(negative_slope=0.2, inplace=True),
    "output_activation": nn.Identity(),
    "bias": False,
    "dropout": 0.5,
}
model_name = f"PENGUIN_n1_z{data_params['redshift']:.0f}_lp{architecture_params['layer_param']}_na{data_params['newton_augmentation']:.1f}"

model_params = {
    "architecture": arch.PENGUIN,
    "model_name": model_name,
    "load_model": True,
    "model_save_path": f"models/{model_name}.pt",
}

loader_params = {
    "batch_size": 1,
    "num_workers": 16,
    "prefetch_factor": 1,
    # "pin_memory": True,
    # "shuffle": True,
    # "drop_last": True,
}

optimizer_params = {
    "lr": 1e-4,
    "betas": (0.5, 0.999),
    "weight_decay": 1e-11,
}

training_params = {
    "epochs": 50,
    "breakout_loss": 1e-2,
    "tol": 0.25,
    "writer_log_path": f"testruns/{model_params['model_name']}_lr{optimizer_params['lr']:.5f}",
    "test_every": 2,
}


if __name__ == "__main__":
    print(f"MULTIPLE_GPUS: {MULTIPLE_GPUS}")
    # print all info from config file
    if VERBOSE:
        print("CONFIGURATION:")
        for dicti in [
            data_params,
            architecture_params,
            model_params,
            loader_params,
            optimizer_params,
            training_params,
        ]:
            for key, value in dicti.items():
                print(f"{key}: {value}")
            print("\n")

    train.train(
        data_params,
        architecture_params,
        model_params,
        loader_params,
        optimizer_params,
        training_params,
    )
