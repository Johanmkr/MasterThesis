""" 
This file is used to test the parallelization of the code, across the GPUs
"""

######### STUFF ###########################################

# stuff

###############################################################

######### IMPORTS ###########################################
import os, sys
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import pandas as pd
from IPython import embed
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from src.data.cube_datasets import SlicedCubeDataset
from src.models.RACOON import RACOON

###############################################################
######### VARIABLES ###########################################
# Data variables
train_test_split = (0.8, 0.2)
batch_size = 256 * 2 * 1  # corresponds to 2/3 cubes
num_workers = 8
redshift = 1.0
stride = 1
train_test_seeds = np.arange(0, 250, 1)
val_seeds = np.arange(1750, 1800, 1)
prefetch_factor = 8
random_seed = 42

# Model variables
input_size = (stride, 256, 256)
layer_param = 64
activation = nn.LeakyReLU(0.2)
output_activation = nn.Sigmoid()
bias = False
dropout = 0.5

# Training variables
lr = 0.001
betas = (0.5, 0.999)
weight_decay = 1e-5
max_epochs = 750
best_loss = 1e10
breakout_loss = 1e-5

# Logging and saving
writer_path = "runs/first"
model_save_path = "models/first.pt"

# Print summary of all variables as one string
# print(f"Variables:\n{pd.DataFrame(globals().items(), columns=['Parameter', 'Value'])}")


###############################################################
######### LOAD DATA ###########################################
def get_data():
    random.seed(random_seed)
    random.shuffle(train_test_seeds)

    array_length = len(train_test_seeds)
    assert (
        abs(sum(train_test_split) - 1.0) < 1e-6
    ), "Train and test split does not sum to 1."
    train_length = int(array_length * train_test_split[0])
    test_length = int(array_length * train_test_split[1])
    train_seeds = train_test_seeds[:train_length]
    test_seeds = train_test_seeds[train_length:]

    # Make datasets
    print("Making datasets...")
    print(f"Training set: {len(train_seeds)} seeds")
    train_dataset = SlicedCubeDataset(
        stride=stride,
        redshift=redshift,
        seeds=train_seeds,
        use_transformations=False,
    )
    print(f"Test set: {len(test_seeds)} seeds")
    test_dataset = SlicedCubeDataset(
        stride=stride,
        redshift=redshift,
        seeds=test_seeds,
        use_transformations=False,
    )
    return train_dataset, test_dataset


###############################################################
######### MODEL ###############################################


def get_model():
    model = RACOON(
        input_size=input_size,
        layer_param=layer_param,
        activation=activation,
        output_activation=output_activation,
        bias=bias,
        dropout=dropout,
    )
    print(f"Model:\n{model}")
    return model


###############################################################
######### GPU stuff ###########################################
def get_gpu_info():
    GPU = torch.cuda.is_available()
    world_size = torch.cuda.device_count()

    print("GPU: ", GPU)
    print("World size: ", world_size)
    return GPU, world_size


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


###############################################################
######### TRAINING (PARALLEL) #################################


def success(output, label):
    tol = 1e-2
    return (abs(output - label) < tol).sum().item()


def evaluate(model, rank, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    correct_guesses = 0
    nr_samples = len(test_loader.dataset)
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["image"], batch["label"]
            images = images.to(rank)
            labels = labels.to(rank)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Print statistics
            test_loss += loss.item()
    return test_loss, correct_guesses, nr_samples


def train(
    rank,
    world_size,
    train_dataset,
    test_dataset,
    model,
    best_loss=best_loss,
    breakout_loss=breakout_loss,
):
    setup(rank, world_size)
    print(f"Rank {rank} starting training...")
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # Make dataloaders
    print("Making dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=train_sampler,
        # shuffle=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=test_sampler,
        # shuffle=True,
        pin_memory=True,
    )

    # Make model distributed
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(
        ddp_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )

    # Writer
    if rank == 0:
        writer = SummaryWriter(writer_path)

    # Train model
    for epoch in range(max_epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        correct_guesses = 0
        total_guesses = 0
        nr_batches = len(train_dataloader)
        print(
            f"\nTraining for epoch {epoch+1} / {max_epochs}..."
        ) if rank == 0 else None
        for i, batch in enumerate(train_dataloader):
            if rank == 0 and (i + 1) % 25 == 0:
                print(f"Batch {i+1} / {nr_batches}")
            images, labels = batch["image"], batch["label"]
            images = images.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            correct_guesses += success(outputs, labels)
            total_guesses += labels.size(0)

        # Calculate rank statistics and send to rank 0
        accuracy_tensor = torch.tensor(
            [correct_guesses, total_guesses], dtype=torch.float
        ).to(rank)
        loss_tensor = torch.tensor([epoch_loss], dtype=torch.float).to(rank)
        reduced_accuracy = [torch.zeros(2).to(rank) for _ in range(world_size)]
        reduced_loss = [torch.zeros(1).to(rank) for _ in range(world_size)]
        dist.all_gather(reduced_loss, loss_tensor)
        dist.all_gather(reduced_accuracy, accuracy_tensor)

        # Evaluate
        test_loss, test_correct_guesses, test_total_guesses = evaluate(
            ddp_model.module, rank, loss_fn, test_dataloader
        )

        test_loss_tensor = torch.tensor([test_loss], dtype=torch.float).to(rank)
        test_accuracy_tensor = torch.tensor(
            [test_correct_guesses, test_total_guesses], dtype=torch.float
        ).to(rank)
        reduced_test_loss = [torch.zeros(1).to(rank) for _ in range(world_size)]
        reduced_test_accuracy = [torch.zeros(2).to(rank) for _ in range(world_size)]
        dist.all_gather(reduced_test_loss, test_loss_tensor)
        dist.all_gather(reduced_test_accuracy, test_accuracy_tensor)

        # Placeholder for early stopping across processes
        should_stop = torch.tensor(False, dtype=torch.bool).to(rank)

        # Calculate and log statistics on rank 0
        if rank == 0:
            # Mean loss and accuracy TRAINING
            mean_loss = torch.stack(reduced_loss).mean().item()
            total_correct_train, total_samples_train = (
                torch.stack(reduced_accuracy).sum(dim=0).tolist()
            )
            mean_accuracy = total_correct_train / total_samples_train

            # Mean loss and accuracy TESTING
            mean_loss_test = torch.stack(reduced_test_loss).mean().item()
            total_correct_test, total_samples_test = (
                torch.stack(reduced_test_accuracy).sum(dim=0).tolist()
            )
            mean_accuracy_test = total_correct_test / total_samples_test

            print(
                f"----------------------\nFinished epoch {epoch+1} / {max_epochs}: \nTrain Loss: {mean_loss:.4e} \nTrain Accuracy: {mean_accuracy*100:.2f}% \nTest Loss: {mean_loss_test:.4e} \nTest Accuracy: {mean_accuracy_test*100:.2f}%\n----------------------"
            )
            # Write to tensorboard
            writer.add_scalars(
                "Loss", {"train": mean_loss, "test": mean_loss_test}, epoch
            )
            writer.add_scalars(
                "Accuracy",
                {"train": mean_accuracy, "test": mean_accuracy_test},
                epoch,
            )
            if epoch % 10 == 0:
                writer.flush()

            # Early stopping condition
            if mean_loss < breakout_loss:
                should_stop = torch.tensor(True, dtype=torch.bool).to(rank)

        # Distribute early stopping across all processes
        dist.broadcast(should_stop, src=0)

        # Save model if mean loss is better than best loss or if early stopping
        if rank == 0:
            if should_stop or mean_loss < best_loss:
                best_loss = mean_loss
                print(f"New best loss found: {best_loss:.4e} Saving model...")

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": mean_loss,
                    },
                    model_save_path,
                )
        # break all processes if early stopping
        if should_stop.item():
            print(f"Rank {rank} breaking...")
            break

        # Synchronize
        dist.barrier()
    print(f"Trained for {epoch+1} epochs") if rank == 0 else None
    print(f"Rank {rank} finished training...")

    # Closing writer and cleaning up
    if rank == 0:
        writer.flush()
        writer.close()

    cleanup()


###############################################################
######### MAIN ################################################


def main():
    train_dataset, test_dataset = get_data()
    model = get_model()
    GPU, world_size = get_gpu_info()
    torch.multiprocessing.spawn(
        train,
        args=(
            world_size,
            train_dataset,
            test_dataset,
            model,
        ),
        nprocs=world_size,
        join=True,
    )


###############################################################

if __name__ == "__main__":
    main()
    # pass
