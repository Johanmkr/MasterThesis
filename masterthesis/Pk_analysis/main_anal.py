import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data as datafile
import architectures
import optuna
from torch.utils.data import DataLoader


class Objective(object):
    def __init__(self, input_size, output_size, max_layers, max_neurons_layers, device,
                 epochs, batch_size, workers):

        self.input_size         = input_size
        self.output_size        = output_size
        self.max_layers         = max_layers
        self.max_neurons_layers = max_neurons_layers
        self.device             = device
        self.epochs             = epochs
        self.batch_size         = batch_size
        self.workers            = workers

    def __call__(self, trial):

        # name of the files that will contain the losses and model weights
        fout   = 'Pk_analysis/losses/loss_%d.txt'%(trial.number)
        fmodel = 'Pk_analysis/models/model_%d.pt'%(trial.number)

        # generate the architecture
        model = architectures.dynamic_model(trial, self.input_size, self.output_size, 
                            self.max_layers, self.max_neurons_layers).to(self.device)

        # get the weight decay and learning rate values
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e0,  log=True)

        # define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                                      weight_decay=wd)

        # get the data
        # train_loader = data.create_dataset('train', self.seed, f_Pk, f_Pk_norm, 
        #                                    f_params, self.batch_size, 
        #                                    shuffle=True, workers=self.workers)
        # valid_loader = data.create_dataset('valid', self.seed, f_Pk, f_Pk_norm, 
        #                                    f_params, self.batch_size, 
        #                                    shuffle=False, workers=self.workers)
        train_data, test_data = datafile.create_data(train_seeds, test_seeds)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.workers)
        
        # train/validate model
        min_valid = 1e40
        for epoch in range(self.epochs):

            # do training
            # train_loss1, train_loss = torch.zeros(len(g)).to(device), 0.0
            # train_loss2, points     = torch.zeros(len(g)).to(device), 0
            model.train()
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            load_len = len(train_loader)
            train_loss = 0
            for i, data in enumerate(train_loader):
                if (i+1) % 100 == 0: print(f"Train Batch {i+1}/{load_len}")
                inputs, labels = data["Pk"], data["label"]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                predictions = model(inputs)
                loss = nn.BCEWithLogitsLoss()(predictions, labels)
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader.dataset)

            # do validation
            test_loss = 0
            model.eval()
            with torch.no_grad():
                load_len = len(test_loader)
                for i, data in enumerate(test_loader):
                    if (i+1) % 100 == 0: print(f"Test Batch {i+1}/{load_len}")
                    inputs, labels = data["Pk"], data["label"]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward
                    predictions = model(inputs)
                    loss = nn.BCEWithLogitsLoss()(predictions, labels)
                    test_loss += loss.item()
                    
            test_loss /= len(test_loader.dataset)

            # save best model if found
            if test_loss<min_valid:  
                min_valid = test_loss
                torch.save(model.state_dict(), fmodel)
            f = open(fout, 'a')
            f.write('%d %.5e %.5e\n'%(epoch, train_loss, test_loss))
            f.close()

            # Handle pruning based on the intermediate value
            # comment out these lines if using prunning
            #trial.report(min_valid, epoch)
            #if trial.should_prune():  raise optuna.exceptions.TrialPruned()

        return min_valid

##################################### INPUT ##########################################
# data parameters
# f_Pk      = 'Pk_galaxies_IllustrisTNG_LH_33_kmax=20.0.npy'
# f_Pk_norm = None
# f_params  = 'latin_hypercube_params.txt' 
# seed      = 1
# train_seeds = np.arange(0,200)
# test_seeds = np.arange(200,250)

train_seeds = np.arange(0,2)
test_seeds = np.arange(2,3)

# architecture parameters
input_size         = 127 #number of bins in Pk
output_size        = 1  #number of parameters to predict (posterior mean + std)
max_layers         = 5
max_neurons_layers = 1000

# training parameters
batch_size = 256 * 10
epochs     = 1000
workers    = 10     #number of cpus to load the data 
# g          = [0,1]  #minimize loss using parameters 0 and 1
# h          = [2,3]  #minimize loss using errors of parameters 0 and 1

# optuna parameters
study_name       = 'Pk_2_params'
n_trials         = 1000 #set to None for infinite
storage          = 'Pk_analysis/params.db'
n_jobs           = 1
n_startup_trials = 20 #random sample the space before using the sampler
######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# create output folders if they dont exist
for fout in ['models', 'losses']:
    if not(os.path.exists("Pk_analysis/"+fout)): os.makedirs("Pk_analysis/"+fout)
    
# define the optuna study and optimize it
objective = Objective(input_size, output_size, max_layers, max_neurons_layers, 
                      device, epochs, batch_size, workers)
sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study = optuna.create_study(study_name=study_name, sampler=sampler)
study.optimize(objective, n_trials, n_jobs=n_jobs)