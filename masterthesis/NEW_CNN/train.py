import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import time
from torch.utils.tensorboard import SummaryWriter

from data import create_data
from architecture import model_o3_err
import train_utils as tutils

"""
    Trial 9 finished with value: 0.0010168541921302676 and parameters: {'hidden': 8, 'dr': 0.15527713188782274, 'optimizer': 'Adam', 'lr': 0.060690469248878935}. Best is trial 9 with value: 0.0010168541921302676.
"""

# GPU control
####################################################################################################
# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
####################################################################################################

# Parameter control
####################################################################################################
# Data
TRAIN_SEEDS = np.arange(0, 200)
TEST_SEEDS = np.arange(200, 250)
BATCH_SIZE = 256
NUM_WORKERS = 4

# Model
MODEL_NAME = "T1"
MODEL_PATH = f"models/{MODEL_NAME}.pt"

# Tunable parameters
HIDDEN = 8  # Hidden layer size
DR = 0.15527713188782274  # Dropout rate
OPTIMIZER = "Adam"  # Optimizer
LR = 0.060690469248878935  # Learning rate
ALPHA = None  # L2 regularization
MOMENTUM = None  # Momentum

# Training
EPOCHS = 100
SAVE_TMP_COUNT = 20
TEST_INTERVAL = 2
LOG_PATH = f"runs/{MODEL_NAME}"

# Make string with information about the parameters:
infostring = f"Model: {MODEL_NAME}\n"
infostring += f"\nhidden={HIDDEN}\ndr={DR}\noptimizer={OPTIMIZER}\nlr={LR}\nalpha={ALPHA}\nmomentum={MOMENTUM}"
infostring += f"\nepochs={EPOCHS}\ntest_interval={TEST_INTERVAL}\nlog_path={LOG_PATH}"
print(infostring)
####################################################################################################


# Create dataloaders
####################################################################################################
train_dataset, test_dataset = create_data(TRAIN_SEEDS, TEST_SEEDS)
# train_sampler = RandomSampler(train_dataset)
# test_sampler = RandomSampler(test_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    # sampler=train_sampler,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    # sampler=test_sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
####################################################################################################

# Create / load state dictionary
####################################################################################################
state = tutils.get_state(MODEL_PATH, load_model=True)
epochs_trained = state["epoch"]

# Create model
####################################################################################################
model = model_o3_err(HIDDEN, DR)
if state["model_state_dict"] is not None:
    model.load_state_dict(state["model_state_dict"])
model.to(device)
loss_fn = nn.BCEWithLogitsLoss()

# Create optimizer
####################################################################################################
if OPTIMIZER.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
elif OPTIMIZER.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
elif OPTIMIZER.lower() == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=ALPHA)
else:
    raise ValueError(
        f"Optimizer {OPTIMIZER} not recognized, must be Adam, SGD or RMSprop"
    )
if state["optimizer_state_dict"] is not None:
    optimizer.load_state_dict(state["optimizer_state_dict"])
####################################################################################################

# Create tensorboard writer
####################################################################################################
writer = SummaryWriter(LOG_PATH)

# Best loss
####################################################################################################
try:
    best_loss = state["best_loss"]
except KeyError:
    best_loss = 1e10

# Training loop
####################################################################################################
for epoch in range(epochs_trained + 1, epochs_trained + EPOCHS + 1):
    # ----------------------- TRAINING --------------------------------
    epoch_start = time.time()
    train_start = time.time()
    (train_loss, train_TP, train_TN, train_FP, train_FN) = tutils.train_one_epoch(
        device=device,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch_nr=epoch,
    )
    train_time = time.time() - train_start

    # Write statistics
    tutils.print_and_write_statistics(
        writer=writer,
        epoch_nr=epoch,
        loss=train_loss,
        TP=train_TP,
        TN=train_TN,
        FP=train_FP,
        FN=train_FN,
        suffix="train",
        time=train_time,
    )

    if epoch % TEST_INTERVAL == 0:
        # ----------------------- TESTING --------------------------------
        test_start = time.time()
        (test_loss, test_TP, test_TN, test_FP, test_FN) = tutils.evaluate(
            device=device,
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
        )
        test_time = time.time() - test_start

        # Write statistics
        tutils.print_and_write_statistics(
            writer=writer,
            epoch_nr=epoch,
            loss=test_loss,
            TP=test_TP,
            TN=test_TN,
            FP=test_FP,
            FN=test_FN,
            suffix="test",
            time=test_time,
        )

        # Save model if loss is better
        best_loss = test_loss if test_loss < best_loss else best_loss
        state["epoch"] = epoch
        state["model_state_dict"] = model.state_dict()
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["train_loss"] = train_loss
        state["test_loss"] = test_loss
        state["best_loss"] = best_loss

        temp_save_path = f"models/{MODEL_NAME}_tmp{state['tmp_save_count']}.pt"

        # Save tmp file
        torch.save(state, temp_save_path)

        # Save master file
        if np.isclose(test_loss, best_loss, rtol=1e-4):
            torch.save(state, MODEL_PATH)
            print(f"Model saved at {MODEL_PATH} with loss {best_loss:.5f}")

        if state["tmp_save_count"] == SAVE_TMP_COUNT:
            state["tmp_save_count"] = 0
        else:
            state["tmp_save_count"] += 1

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch} took {epoch_time:.2f} seconds")
    writer.flush()
writer.close()
####################################################################################################
