import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data3D as datafile3D
import architectures
from torch.utils.data import DataLoader
import optuna

################################### INPUT ############################################
# data parameters
# f_Pk      = 'Pk_galaxies_SIMBA_LH_33_kmax=20.0.npy'        #file with Pk
# f_Pk_norm = 'Pk_galaxies_IllustrisTNG_LH_33_kmax=20.0.npy' #file with Pk to normalize Pk
# f_params  = 'latin_hypercube_params.txt'                   #file with parameters
# seed      = 1                                    #seed to split data in train/valid/test
# mode      = 'all'     #'train','valid','test' or 'all'

# architecture parameters
input_size = 127  # number of bins in Pk
output_size = 1  # number of parameters to predict (posterior mean + std)

# training parameters
# batch_size = 32
# workers    = 1
# g          = [0,1]
# h          = [2,3]

# optuna parameters
study_name = "3DPk_FCNN_BC"
storage = "sqlite:///TPE.db"

# output file name
fout_S1 = "3D_S1_results.txt"
fout_S2 = "3D_S2_results.txt"
######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device("cuda")
else:
    print("CUDA Not Available")
    device = torch.device("cpu")

# load the optuna study
study = optuna.load_study(study_name=study_name, storage=storage)

# get the scores of the study trials
values = np.zeros(len(study.trials))
completed = 0
for i, t in enumerate(study.trials):
    values[i] = t.value
    if t.value is not None:
        completed += 1

# get the info of the best trial
indexes = np.argsort(values)
for i in [0]:  # choose the best-model here, e.g. [0], or [1]
    trial = study.trials[indexes[i]]
    print("\nTrial number {}".format(trial.number))
    print("Value: %.5e" % trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    n_layers = trial.params["n_layers"]
    lr = trial.params["lr"]
    wd = trial.params["wd"]
    hidden = np.zeros(n_layers, dtype=np.int32)
    dr = np.zeros(n_layers, dtype=np.float32)
    for i in range(n_layers):
        hidden[i] = trial.params["n_units_l%d" % i]
        dr[i] = trial.params["dropout_l%d" % i]
    fmodel = "Pk_analysis/3Dmodels/model_%d.pt" % trial.number

# generate the architecture
model = architectures.dynamic_model2(input_size, output_size, n_layers, hidden, dr)
model.to(device)

# load best-model, if it exists
print("Loading model...")
if os.path.exists(fmodel):
    model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
else:
    raise Exception("model doesnt exists!!!")

# get the data
# test_loader = data.create_dataset(mode, seed, f_Pk, f_Pk_norm, f_params, batch_size, shuffle=False, workers=workers)
# test_points = 0
# for x,y in test_loader:  test_points += x.shape[0]

# define the matrix containing the true and predicted value of the parameters + errors
# params  = len(g)

(mean, variance) = np.load("mean_var.npy")
S1_data = datafile3D.ScaledData(
    seeds=np.arange(200, 250), train=False, mean=mean, variance=variance
)
S2_data = datafile3D.ScaledData(
    seeds=np.arange(250, 300), train=False, mean=mean, variance=variance
)

S1_loader = DataLoader(
    S1_data, batch_size=2 * 50, shuffle=False, num_workers=24, pin_memory=True
)
S2_loader = DataLoader(
    S2_data, batch_size=2 * 50, shuffle=False, num_workers=24, pin_memory=True
)

loss_fn = nn.BCEWithLogitsLoss(reduction="none")
outfunc = nn.Sigmoid()

# save_categoreis = ["predictions", "true_values", "loss"]

# test the model
# test_loss1, test_loss = torch.zeros(len(g)).to(device), 0.0
# test_loss2, points    = torch.zeros(len(g)).to(device), 0
model.eval()
for loader, fout in zip([S1_loader, S2_loader], [fout_S1, fout_S2]):
    results = np.zeros((len(loader.dataset), 4))
    for data in loader:
        with torch.no_grad():
            Pk = data["Pk"].to(device)
            label = data["label"].to(device)
            score = model(Pk)
            loss = loss_fn(score, label)
            pred = outfunc(score)
            results[:, 0] = score.cpu().numpy()[:, 0]
            results[:, 1] = pred.cpu().numpy()[:, 0]
            results[:, 2] = label.cpu().numpy()[:, 0]
            results[:, 3] = loss.cpu().numpy()[:, 0]
    np.savetxt(fout, results, header="score prediction true_values loss")


# for x, y in test_loader:
#     with torch.no_grad():
#         bs    = x.shape[0]         #batch size
#         x     = x.to(device)       #maps
#         y     = y.to(device)[:,g]  #parameters
#         p     = model(x)           #NN output
#         y_NN  = p[:,g]             #posterior mean
#         e_NN  = p[:,h]             #posterior std
#         loss1 = torch.mean((y_NN - y)**2,                axis=0)
#         loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
#         loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
#         test_loss1 += loss1*bs
#         test_loss2 += loss2*bs
#         results[points:points+bs,0*params:1*params] = y.cpu().numpy()
#         results[points:points+bs,1*params:2*params] = y_NN.cpu().numpy()
#         results[points:points+bs,2*params:3*params] = e_NN.cpu().numpy()
#         points     += bs
# test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
# test_loss = torch.mean(test_loss).item()
# print('Test loss:', test_loss)

# denormalize results here
# minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[g]
# maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[g]
# results[:,0*params:1*params] = results[:,0*params:1*params]*(maximum-minimum)+minimum
# results[:,1*params:2*params] = results[:,1*params:2*params]*(maximum-minimum)+minimum
# results[:,2*params:3*params] = results[:,2*params:3*params]*(maximum-minimum)


# # save results to file
# np.savetxt(fout_T1, results_T1)
# np.savetxt(fout_T2, results_T2)
