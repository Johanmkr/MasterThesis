VERBOSE = True
MULTIPLE_GPUS = True

import numpy as np
import torch.nn as nn

import architectures as arch

if MULTIPLE_GPUS:
    import train_multigpu as train
else:
    import train_singlegpu as train

########################## MOST IMPORTANT PARAMS ############################
train_test_division_at_seed = 100
total_seeds = 120
newton_augmentation = (
    0.0  # must be one for serious training, change for testing pipeline.
)

architecture = "PENGUIN"
layer_param = 32

lr = 1e-4
betas = (0.5, 0.999)
weight_decay = 1e-11

epochs = 5


########################## MODEL NAME and DATA PATH ##########################
datapath = "/mn/stornext/d10/data/johanmkr/simulations/data_z1/data_z1.h5"
dataname = datapath.split("/")[-1].split(".")[0]
model_name = "code_test"
# model_name = f"TEST_{dataname}_{architecture}_lp{layer_param}_na{newton_augmentation}"


# Params
data_params = {
    "train_seeds": np.arange(0, train_test_division_at_seed, 1),
    "test_seeds": np.arange(train_test_division_at_seed, total_seeds, 1),
    "newton_augmentation": 0.0,  # must be one for serious training, change for testing pipeline.
    "datapath": datapath,
}

architecture_params = {
    "input_size": (1, 256, 256),
    "layer_param": layer_param,
    "activation": nn.ReLU(),
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
    "batch_size": 1,
    "num_workers": 16,
    "prefetch_factor": 1,
}

optimizer_params = {
    "lr": lr,
    "betas": betas,
    "weight_decay": weight_decay,
}

training_params = {
    "epochs": epochs,
    "breakout_loss": 1e-4,
    "writer_log_path": f"testruns/{model_name}_lr{lr:.5f}",
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
