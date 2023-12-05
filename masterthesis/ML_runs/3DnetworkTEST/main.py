######### GLOBAL IMPORTS #######################################
import sys, os
import torch
from torch.utils.tensorboard import SummaryWriter


######### ADD PARENT DIRECTORY TO PATH #########################
# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)


######### LOCAL IMPORTS ########################################
from src.data.whole_cube_dataset import make_whole_dataset
from src.models.SLOTH import SLOTH
from src.models.train_model import ModelTrainer


######### IMPORT CONFIG FILE ####################################
import config_3Dtest as cfg


######### GPU STUFF ###########################################
if cfg.VERBOSE:
    print("Checking for GPU...")
GPU = torch.cuda.is_available()
GPU = False
device = torch.device("cuda:0" if GPU else "cpu")
print("GPU: ", GPU)
print("Device: ", device)
# cudnn.benchmark = True
if cfg.VERBOSE:
    print("GPU check complete.\n\n")


######### DATA ###############################################
if cfg.VERBOSE:
    print("Loading data...")
train_loader, test_loader, val_loader = make_whole_dataset(**cfg.DATA_PARAMS)
if cfg.VERBOSE:
    print("Data loaded.\n\n")


######### MODEL ###############################################
if cfg.VERBOSE:
    print("Loading model...")
model = cfg.MODEL
if cfg.VERBOSE:
    print("Model loaded.\n\n")


######### SUMMARY ###############################################
if cfg.VERBOSE:
    print("Printing summary")
    print(f"Model: {model}")


######### OPTIMIZER ###############################################
optimizer = cfg.OPTIMIZER
loss_fn = cfg.LOSS_FN

######### TRAINING ###############################################
writer = SummaryWriter("runs/3Dtest_script")
trainer = ModelTrainer(
    model,
    optimizer,
    loss_fn,
    device,
    verbose=cfg.VERBOSE,
    writer=writer,
    save_path="saved_models/3Dtest_script",
)

# Train 2 epoch
trainer.train_model(
    train_loader,
    test_loader,
    2,
)

######### OVERFIT ###############################################
# overfit_model(
#     model,
#     optimizer,
#     loss_fn,
#     train_loaders[0],
#     device,
#     verbose=True,
#     tol=1e-1,
# )
