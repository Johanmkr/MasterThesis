VERBOSE = True
MULTIPLE_GPUS = True

import numpy as np
import torch
import torch.nn as nn
import sys
import yaml
from datetime import datetime

import architectures as arch

if MULTIPLE_GPUS:
    import train_multigpu as train
else:
    import train_singlegpu as train

########################## MOST IMPORTANT PARAMS ############################

configuration_file = f"configurations/{str(sys.argv[1])}.yaml"

with open(configuration_file, "r") as f:
    conf = yaml.safe_load(f)

architecture = str(conf["model_params"]["architecture"])
layer_param = int(conf["model_params"]["layer_param"])
dropout = float(conf["model_params"]["dropout"])
name = str(conf["model_params"]["name"])

datafile = str(conf["data_params"]["datafile"])
nr_train_seeds = int(conf["data_params"]["train_seeds"])
nr_test_seeds = int(conf["data_params"]["test_seeds"])
newton_augmentation = float(conf["data_params"]["newton_augmentation"])

epochs = int(conf["train_params"]["epochs"])
lr = float(conf["train_params"]["lr"])


########################## MODEL NAME and DATA PATH ##########################
dataname = datafile.split("/")[-1].split(".")[0]
# model_name = "code_test"
model_name = f"{name}_{dataname}_{architecture}_lp{layer_param}_do{dropout}_na{newton_augmentation}_ts{nr_train_seeds}"


# Params
data_params = {
    "train_seeds": np.arange(0, nr_train_seeds, 1),
    "test_seeds": np.arange(nr_train_seeds, nr_train_seeds + nr_test_seeds, 1),
    "newton_augmentation": newton_augmentation,
    "datapath": datafile,
}

architecture_params = {
    "input_size": (1, 256, 256),
    "layer_param": layer_param,
    "activation": nn.LeakyReLU(negative_slope=0.2),
    "output_activation": nn.Identity(),
    "bias": False,
    "dropout": dropout,
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
    "batch_size": 5 if layer_param <= 32 else 2,
    "num_workers": 4 * torch.cuda.device_count(),
    "prefetch_factor": 2,
}

optimizer_params = {
    "lr": lr,
    "betas": (0.5, 0.999),
    "weight_decay": 1e-11,
}

training_params = {
    "epochs": epochs,
    "writer_log_path": f"longruns/{model_name}_lr{lr}",
    "test_every": 2,
}


if __name__ == "__main__":
    infostring = "\n------------------------------------------------------------\n"
    infostring += f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
    infostring += f"MULTIPLE_GPUS: {MULTIPLE_GPUS}\n"
    # print all info from config file
    if VERBOSE:
        infostring += f"Configuration file: {configuration_file}\n\n"
        infostring += "CONFIGURATION:\n"
        for dicti in [
            data_params,
            architecture_params,
            model_params,
            loader_params,
            optimizer_params,
            training_params,
        ]:
            for key, value in dicti.items():
                infostring += f"{key}: {value}\n"
            infostring += f"\n"
    infostring += "------------------------------------------------------------\n"
    infostring += f"\n\n\n"
    print(infostring)

    with open(f"dummy_txt_logs/{model_name}.txt", "a") as f:
        f.write(infostring)

    train.train(
        data_params,
        architecture_params,
        model_params,
        loader_params,
        optimizer_params,
        training_params,
    )
