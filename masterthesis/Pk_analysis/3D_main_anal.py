import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data3D as datafile3D
import architectures
import optuna
from torch.utils.data import DataLoader
from tqdm import trange


class Objective(object):
    def __init__(
        self,
        input_size,
        output_size,
        max_layers,
        max_neurons_layers,
        device,
        epochs,
        train_loader,
        test_loader,
    ):

        self.input_size = input_size
        self.output_size = output_size
        self.max_layers = max_layers
        self.max_neurons_layers = max_neurons_layers
        self.device = device
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

    def __call__(self, trial):

        # name of the files that will contain the losses and model weights
        fout = "Pk_analysis/3Dlosses/loss_%d.txt" % (trial.number)
        fmodel = "Pk_analysis/3Dmodels/model_%d.pt" % (trial.number)

        # generate the architecture
        model = architectures.dynamic_model(
            trial,
            self.input_size,
            self.output_size,
            self.max_layers,
            self.max_neurons_layers,
        ).to(self.device)

        loss_fn = nn.BCEWithLogitsLoss()
        # get the weight decay and learning rate values
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e0, log=True)

        # define the optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd
        )

        # get the data
        # train_loader = data.create_dataset('train', self.seed, f_Pk, f_Pk_norm,
        #                                    f_params, self.batch_size,
        #                                    shuffle=True, workers=self.workers)
        # valid_loader = data.create_dataset('valid', self.seed, f_Pk, f_Pk_norm,
        #                                    f_params, self.batch_size,
        #                                    shuffle=False, workers=self.workers)

        train_length = len(self.train_loader.dataset)
        test_length = len(self.test_loader.dataset)

        print(f"\nTrial {trial.number}")

        # train/validate model
        min_valid = 1e40

        # from IPython import embed; embed()

        # Do everything on the GPU since entire training loader can fit in GPU memory as one single batch. No need for batch iteration and data transfer between CPU and GPU.

        # Transfer to gpu
        for data in self.train_loader:
            train_pk_gpu = data["Pk"].to(self.device)
            train_label_gpu = data["label"].to(self.device)

        for data in self.test_loader:
            test_pk_gpu = data["Pk"].to(self.device)
            test_label_gpu = data["label"].to(self.device)

        for epoch in trange(self.epochs, desc="Epochs"):
            model.train()
            train_loss = 0

            # Shuffle training data
            idx = torch.randperm(train_pk_gpu.size(0))
            train_pk_gpu = train_pk_gpu[idx]
            train_label_gpu = train_label_gpu[idx]

            # Forward
            predictions = model(train_pk_gpu)
            loss = loss_fn(predictions, train_label_gpu)
            train_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train_loss /= train_length

            # Testing
            test_loss = 0
            model.eval()
            with torch.no_grad():
                predictions = model(test_pk_gpu)
                loss = loss_fn(predictions, test_label_gpu)
                test_loss += loss.item()
            # test_loss /= test_length

            # for epoch in range(self.epochs):

            #     # do training
            #     # train_loss1, train_loss = torch.zeros(len(g)).to(device), 0.0
            #     # train_loss2, points     = torch.zeros(len(g)).to(device), 0
            #     model.train()
            #     print(f"Epoch {epoch+1}/{self.epochs}")
            #     load_len = len(train_loader)
            #     train_loss = 0
            #     for i, data in enumerate(train_loader):
            #         # if (i+1) % 100 == 0: print(f"Train Batch {i+1}/{load_len}")
            #         inputs, labels = data["Pk"], data["label"]
            #         inputs = inputs.to(self.device)
            #         labels = labels.to(self.device)

            #         # Forward
            #         predictions = model(inputs)
            #         loss = nn.BCEWithLogitsLoss()(predictions, labels)
            #         train_loss += loss.item()

            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
            #     train_loss /= len(train_loader.dataset)

            # # do validation
            # test_loss = 0
            # model.eval()
            # with torch.no_grad():
            #     load_len = len(test_loader)
            #     for i, data in enumerate(test_loader):
            #         # if (i+1) % 100 == 0: print(f"Test Batch {i+1}/{load_len}")
            #         inputs, labels = data["Pk"], data["label"]
            #         inputs = inputs.to(self.device)
            #         labels = labels.to(self.device)

            #         # Forward
            #         predictions = model(inputs)
            #         loss = nn.BCEWithLogitsLoss()(predictions, labels)
            #         test_loss += loss.item()

            # test_loss /= len(test_loader.dataset)

            # save best model if found
            if test_loss < min_valid:
                min_valid = test_loss
                torch.save(model.state_dict(), fmodel)
            f = open(fout, "a")
            f.write("%d %.5e %.5e\n" % (epoch, train_loss, test_loss))
            f.close()

            # Handle pruning based on the intermediate value
            # comment out these lines if using prunning
            # trial.report(min_valid, epoch)
            # if trial.should_prune():  raise optuna.exceptions.TrialPruned()

        return min_valid


##################################### INPUT ##########################################
# data parameters
# f_Pk      = 'Pk_galaxies_IllustrisTNG_LH_33_kmax=20.0.npy'
# f_Pk_norm = None
# f_params  = 'latin_hypercube_params.txt'
# seed      = 1
# train_seeds = np.arange(0,200)
# test_seeds = np.arange(200,250)

train_seeds = np.arange(0, 200)
test_seeds = np.arange(200, 250)

# architecture parameters
input_size = 127  # number of bins in Pk
output_size = 1  # number of parameters to predict (posterior mean + std)
max_layers = 5
max_neurons_layers = 1000

# training parameters
# batch_size = 256 * 3 * 2 * 50
epochs = 150
workers = 24  # number of cpus to load the data
# g          = [0,1]  #minimize loss using parameters 0 and 1
# h          = [2,3]  #minimize loss using errors of parameters 0 and 1

# optuna parameters
study_name = "3DPk_FCNN_BC"
n_trials = 500  # set to None for infinite
storage = "sqlite:///TPE.db"
n_jobs = 1
n_startup_trials = 20  # random sample the space before using the sampler
######################################################################################
train_bs = 2 * len(train_seeds)
test_bs = 2 * len(test_seeds)


train_data, test_data = datafile3D.create_data(train_seeds, test_seeds)
train_loader = DataLoader(
    train_data, batch_size=train_bs, shuffle=True, num_workers=workers, pin_memory=True
)
test_loader = DataLoader(
    test_data, batch_size=test_bs, shuffle=False, num_workers=workers, pin_memory=True
)


# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device("cuda")
else:
    print("CUDA Not Available")
    device = torch.device("cpu")

# create output folders if they dont exist
for outpath in ["3Dmodels", "3Dlosses"]:
    if not (os.path.exists("Pk_analysis/" + outpath)):
        os.makedirs("Pk_analysis/" + outpath)

# define the optuna study and optimize it
objective = Objective(
    input_size,
    output_size,
    max_layers,
    max_neurons_layers,
    device,
    epochs,
    train_loader,
    test_loader,
)
sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study = optuna.create_study(
    study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True
)
study.optimize(objective, n_trials, n_jobs=n_jobs)
