VERBOSE = True
MULTIPLE_GPUS = False

import numpy as np
import torch
import torch.nn as nn
import sys, os
import yaml
from datetime import datetime

import architectures as arch

if MULTIPLE_GPUS:
    import train_multigpu as train
else:
    import train_singlegpu as train

########################## MOST IMPORTANT PARAMS ############################
try:
    configuration_file = f"configurations/{str(sys.argv[1])}"
    configuration_file = configuration_file.replace(".yaml", "") + ".yaml"
    if not os.path.exists(configuration_file):
        raise FileNotFoundError
except FileNotFoundError:
    configuration_file = f"{str(sys.argv[1])}"
    configuration_file = configuration_file.replace(".yaml", "") + ".yaml"

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
log_dir = str(conf["train_params"]["log_dir"])
if not os.path.exists(log_dir):
    raise FileNotFoundError
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
    "hidden": layer_param,
    "dr": dropout,
    "channels": 1,
}

model_params = {
    "architecture": arch.model_o3_err,
    "model_name": model_name,
    "load_model": True,
    "model_save_path": f"cnn_storage/models/{model_name}.pt",
}

loader_params = {
    "batch_size": 150,
    "num_workers": torch.cuda.device_count(),
    "prefetch_factor": 2,
}

optimizer_params = {
    "lr": lr,
    "betas": (0.5, 0.999),
    "weight_decay": 1e-8,
}

training_params = {
    "epochs": epochs,
    "writer_log_path": f"{log_dir}/{model_name}_lr{lr}",
    "save_tmp_every": 10,
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
