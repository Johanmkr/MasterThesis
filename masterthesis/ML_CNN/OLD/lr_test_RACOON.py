VERBOSE = True
MULTIPLE_GPUS = True

import numpy as np
import torch.nn as nn
import sys

import architectures as arch

if MULTIPLE_GPUS:
    import train_multigpu as train
else:
    import train_singlegpu as train

########################## MOST IMPORTANT PARAMS ############################
train_test_division_at_seed = 50
total_seeds = 60
newton_augmentation = (
    1.0  # must be one for serious training, change for testing pipeline.
)

architecture = "RACOON"
layer_param = 64

lr = float(sys.argv[1])
betas = (0.5, 0.9999)
weight_decay = 1e-11

epochs = 25


########################## MODEL NAME and DATA PATH ##########################
datapath = "/mn/stornext/d10/data/johanmkr/simulations/data_z1/data_z1.h5"
dataname = datapath.split("/")[-1].split(".")[0]
# model_name = "code_test"
model_name = f"TEST_main_data_LRELU02_lr{lr}_{dataname}_{architecture}_lp{layer_param}_na{newton_augmentation}"


# Params
data_params = {
    "train_seeds": np.arange(0, train_test_division_at_seed, 1),
    "test_seeds": np.arange(train_test_division_at_seed, total_seeds, 1),
    "newton_augmentation": newton_augmentation,
    "datapath": datapath,
}

architecture_params = {
    "input_size": (1, 256, 256),
    "layer_param": layer_param,
    "activation": nn.LeakyReLU(negative_slope=0.2),
    "output_activation": nn.Identity(),
    "bias": False,
    "dropout": 0.5,
}

model_params = {
    "architecture": arch.PENGUIN
    if architecture.lower() in ["penguin", "p"]
    else arch.RACOON,
    "model_name": model_name,
    "load_model": True,
    "model_save_path": f"models/{model_name}.pt",
}

loader_params = {
    "batch_size": 2,
    "num_workers": 32,
    "prefetch_factor": 2,
}

optimizer_params = {
    "lr": lr,
    "betas": betas,
    "weight_decay": weight_decay,
}

training_params = {
    "epochs": epochs,
    "breakout_loss": 1e-4,
    "writer_log_path": f"lr_tests_main_data/{model_name}_lr{lr}",
    "test_every": 1,
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
