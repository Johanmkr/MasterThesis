import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from data import create_data
from architecture import model_o3_err

# Check if CUDA is available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generate data. This is a simple example, so we will use a simple dataset.
# Generate the dataset.
train_seeds = np.arange(0, 50)
test_seeds = np.arange(50, 70)
train_dataset, test_dataset = create_data(train_seeds, test_seeds)

"""
Trial 3 finished with value: 0.10217975080013275 and parameters: {'hidden': 10, 'dr': 0.010887116302778088, 'optimizer': 'Adam', 'lr': 0.00017401947422863024}. Best is trial 3 with value: 0.10217975080013275.
"""


def objective(trial):
    # Architecture parameters
    hidden = trial.suggest_int("hidden", 6, 12)
    dr = trial.suggest_float("dr", 0.0, 0.9)

    # Generate the model.
    model = model_o3_err(hidden, dr)
    model = model.to(device)

    # Optimizer parameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        alpha = trial.suggest_float("alpha", 0.1, 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 1.0)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Generate the dataloader.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Print progress info
    print(f"\n\nTraining with the following parameters:")
    print(f"hidden: {hidden}\ndr: {dr}\noptimizer: {optimizer_name}\nlr: {lr}\n\n")

    # Training of the model.
    for epoch in range(25):
        model.train()
        print(f"\nEpoch {epoch+1}/25")
        load_len = len(train_loader)
        for i, data in enumerate(train_loader):
            if (i + 1) % 50 == 0:
                print(f"Train Batch {i+1}/{load_len}")
            # Load data
            inputs, labels = data["image"], data["label"]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(inputs)
            loss = nn.BCEWithLogitsLoss()(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optimization step
            optimizer.step()

    print("Training done")
    print("Evaluating model...")
    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        load_len = len(test_loader)
        for i, data in enumerate(test_loader):
            if (i + 1) % 50 == 0:
                print(f"Test Batch {i+1}/{load_len}")
            inputs, labels = data["image"], data["label"]
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)
            test_loss += nn.BCEWithLogitsLoss()(predictions, labels)

    test_loss /= len(test_loader.dataset)

    return test_loss


if __name__ == "__main__":
    "Set up study and optimize"
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=45)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save study
    best_params = trial.params
    with open("best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    study.trials_dataframe().to_csv("study.csv")
    study.trials_dataframe().to_pickle("study.pkl")
