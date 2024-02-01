import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import time
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from IPython import embed

import data

import train_utils as tutils

mp.set_sharing_strategy("file_descriptor")

GPU = torch.cuda.is_available()
world_size = torch.cuda.device_count()

output_func = nn.Sigmoid()


######################### DDP functions #########################
def setup(rank:int, world_size:int):
    # Setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def model_init(rank: int, world_size:int, model:torch.nn.Module) -> torch.nn.Module:
    """Initialized the sub-models created in each sub-process. 

    Args:
        rank (int): Process ID
        world_size (int): Number of processess.
        model (torch.nn.Module): Main model

    Returns:
        torch.nn.Module: Submodel for the sub-process specified.
    """
    # setup the process group
    setup(rank, world_size)

    # Explicitly setting seed to make sure that models created in two processes start from same random weights and biases.
    torch.manual_seed(42)

    # create model and move it to GPU with id rank
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    return ddp_model


def cleanup():
    # Cleans up the creates sub-processes
    dist.destroy_process_group()


##################################################################


def create_data_loaders(
    rank: int,
    world_size: int,
    train_dataset: torch.utils.data.dataset.Dataset,
    test_dataset: torch.utils.data.dataset.Dataset,
    batch_size: int,
    num_workers:int=0,
    prefetch_factor:int=2,
    pin_memory:bool=False,
    shuffle:bool=True,
    drop_last:bool=False,
):
    """Create dataloader, via samplers, from the original dataset objects. Each process needs their own dataloader. Each process initialize a sampler, given a rank, which samples its share of indices that make up the dataset. A dataloader object is then created from the sampler. This ensures that different processes train/test on different parts of the main dataset. 

    Args:
        rank (int): Process ID
        world_size (int): Number of processes.
        train_dataset (torch.utils.data.dataset.Dataset): Training data
        test_dataset (torch.utils.data.dataset.Dataset): Testing data
        batch_size (int): Size of batch used simultaneously in training/testing logic
        num_workers (int, optional): Number of workers/processors to pre-load data. Defaults to 0.
        prefetch_factor (int, optional): Number of batches each worker pre-loads. Defaults to 2.
        pin_memory (bool, optional): Pins memory in RAM. Defaults to False.
        shuffle (bool, optional): Shuffle when initialising sampler. Defaults to True.
        drop_last (bool, optional): Drop last batch so their size is uniform. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=train_sampler,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=test_sampler,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, train_sampler, test_sampler


def worker(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.dataset.Dataset,
    test_dataset: torch.utils.data.dataset.Dataset,
    loader_params: dict,
    optimizer_params: dict,
    training_params: dict,
    state: dict,
) -> None:
    """
    Main worker proccess used for training and testing the neural network. Each subprocess must initialize its own worker. 

    Logic: #TODO Check whether the optimizer and loss function need to be initialized for each sub-process. 
        (1) Copy/initialize:        - Model on the specific device/sub-process
                                    - Train and test loaders for each sub-process to manage local training.
                                    - Optimizer instances for each sub-process.
                                    - Loss-function instance.
                                    - If the model is laoded, the above need to be initalized from the loaded state dictionary.
                                    - Tensorboard instance for logging progress. 
        (2) Loop over the number of epochs specified in training params. 
            (2a) Training:
                    - Train one epoch.
                    - Gather statistics and send to master rank.
                    - Calculate and log relevant statistics (master rank only).
            (2b) Testing:
                    - Evaluate on the testing data. 
                    - Gather statistics and send to master rank. 
                    - Calculate and log relevant statistics (master rank only)
                    - Save model(s) (master rank only).
            (2c) Synchronize processes. 
        (3) Ending training and cleaning up.



    Args:
        rank (int): Device or the rank of the specific proccess.
        world_size (int): Maximum number of proccesses for multi GPU training.
        model (torch.nn.Module): The model on which we train.
        train_dataset (torch.utils.data.dataset.Dataset): Dataset object with the training data.
        test_dataset (torch.utils.data.dataset.Dataset): Dataset object with the testing data.
        loader_params (dict): Parameters to initialise the dataloaders for each sub-proccess.
        optimizer_params (dict): Parameters to initialize the optimizers for each sub-proccess. 
        training_params (dict): Parameters to control the training proccess. 
        state (dict): Dictionary to hold the state parameter of the module for easy saving and loading. 
    """

    ########################
    #   (1) Copy/initialise.
    ########################

    # Initialize model
    ddp_model = model_init(rank, world_size, model)

    # Initialise dataloaders
    train_loader, test_loader, train_sampler, test_sampler = create_data_loaders(
        rank,
        world_size,
        train_dataset,
        test_dataset,
        **loader_params,
    )

    # Initialise optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(), **optimizer_params)
    if state["optimizer_state_dict"] is not None: # Initial new optimizer of not load
        optimizer.load_state_dict(state["optimizer_state_dict"])
    loss_fn = nn.BCEWithLogitsLoss()

    # Set epochs
    epochs_trained = state["epoch"] # Starting to train from this
    max_epochs = epochs_trained + training_params["epochs"] # Last epoch to be trained on current session

    # Tensorboard
    if rank == 0:
        writer = SummaryWriter(training_params["writer_log_path"])
        if not state["model_information_written"]:
            writer.add_graph(model, torch.ones((1, 1, 256, 256)).to(rank))
            state["model_information_written"] = True

    # Set best loss
    try:
        best_loss = state["best_loss"]
    except KeyError:
        best_loss = 1e10

    ####################################################################
    #   (2) Loop over the number of epochs specified in training params. 
    ####################################################################

    for epoch in range(epochs_trained + 1, max_epochs + 1):
        epoch_total_time_start = time.time() if rank == 0 else None

        #-----------------
        #   (2a) Training. 
        #-----------------

        # Timing
        train_sampler.set_epoch(epoch)
        epoch_train_start_time = time.time() if rank == 0 else None
        
        # Train for one epoch
        (
            local_train_loss,
            local_train_TP,
            local_train_TN,
            local_train_FP,
            local_train_FN,
        ) = tutils.train_one_epoch_cube_version(
            rank,
            ddp_model,
            train_loader,
            optimizer,
            loss_fn,
            epoch,
        )
        epoch_train_end_time = time.time() if rank == 0 else None

        # Gather statistics and send to rank 0
        train_metrics = torch.tensor(
            [
                local_train_loss,
                local_train_TP,
                local_train_TN,
                local_train_FP,
                local_train_FN,
            ],
            dtype=torch.float32,
        ).to(rank)

        # Empty tensor to contain the metrics from all processes
        reduced_train_metrics = [
            torch.zeros_like(train_metrics) for _ in range(world_size)
        ]

        # Gather all the metrics onto the newly created tensor
        dist.all_gather(reduced_train_metrics, train_metrics)

        # Calculate and log statistics on master rank
        if rank == 0:
            # Calculate
            (
                total_mean_train_loss,
                total_train_TP,
                total_train_TN,
                total_train_FP,
                total_train_FN,
            ) = (torch.stack(reduced_train_metrics).sum(dim=0)).tolist()
            mean_train_loss = total_mean_train_loss / world_size

            # Log
            tutils.print_and_write_statistics(
                writer=writer,
                epoch_nr=epoch,
                loss=mean_train_loss,
                TP=total_train_TP,
                TN=total_train_TN,
                FP=total_train_FP,
                FN=total_train_FN,
                suffix="train",
                time=epoch_train_end_time - epoch_train_start_time,
            )

        #----------------
        #   (2b) Testing. 
        #----------------
        
        # Test only for every few epochs as set in training params
        if epoch % training_params["test_every"] == 0 or epoch == max_epochs:

            # Time
            epoch_test_start_time = time.time() if rank == 0 else None
            
            # Evaluate
            (
                local_test_loss,
                local_test_TP,
                local_test_TN,
                local_test_FP,
                local_test_FN,
            ) = tutils.evaluate_cube_version(
                rank,
                ddp_model,
                loss_fn,
                test_loader,
            )
            epoch_test_end_time = time.time() if rank == 0 else None

            # Gather statistics and send to master rank
            test_metrics = torch.tensor(
                [
                    local_test_loss,
                    local_test_TP,
                    local_test_TN,
                    local_test_FP,
                    local_test_FN,
                ],
                dtype=torch.float32,
            ).to(rank)
            reduced_test_metrics = [
                torch.zeros_like(test_metrics) for _ in range(world_size)
            ]

            dist.all_gather(reduced_test_metrics, test_metrics)

            # Calculate and log statistics on master rank
            if rank == 0:
                # Calculate
                (
                    total_mean_test_loss,
                    total_test_TP,
                    total_test_TN,
                    total_test_FP,
                    total_test_FN,
                ) = (torch.stack(reduced_test_metrics).sum(dim=0)).tolist()
                mean_test_loss = total_mean_test_loss / world_size

                # Log
                tutils.print_and_write_statistics(
                    writer=writer,
                    epoch_nr=epoch,
                    loss=mean_test_loss,
                    TP=total_test_TP,
                    TN=total_test_TN,
                    FP=total_test_FP,
                    FN=total_test_FN,
                    suffix="test",
                    time=epoch_test_end_time - epoch_test_start_time,
                )

                # Save one model with the current epoch (copy) and overwrite the master save
                best_loss = mean_test_loss
                state["epoch"] = epoch
                state["model_state_dict"] = model.state_dict()
                state["optimizer_state_dict"] = optimizer.state_dict()
                state["train_loss"] = mean_train_loss
                state["test_loss"] = mean_test_loss
                state["best_loss"] = best_loss
                epoch_savepath = (
                    state[f"model_save_path"].split("/")[0]
                    + "/"
                    + "TMP_"
                    + state[f"model_save_path"].split("/")[1].replace(".pt", "")
                    + f"_epoch{state['epoch']}.pt"
                )

                # Save a new copy of the model for the current epoch
                torch.save(
                    state,
                    epoch_savepath,
                )

                # Save/overwrite the current master model 
                torch.save(
                    state,
                    state["model_save_path"],
                )
                print(f"Saved model to {epoch_savepath}")

        #--------------------
        #   (2b) Synchronize. 
        #--------------------
        if rank == 0:
            epoch_total_time_end = time.time()

            print(
                f"Time elapsed for epoch: {epoch_total_time_end - epoch_total_time_start:.2f} s\n"
            )
        
        # Synchronize all processes
        dist.barrier()

    ##############################
    #   (2) Ening and cleaning up.
    ##############################
    print(f"Trained for {epoch} epochs") if rank == 0 else None
    print(f"Rank {rank} finished training")

    # Closing writer and cleaning up
    if rank == 0:
        writer.flush()
        writer.close()

    cleanup()


def train(
    data_params: dict,
    architecture_params: dict,
    model_params: dict,
    loader_params: dict,
    optimizer_params: dict,
    training_params: dict,
) -> None:
    """Main training function used to initialize datasets and (loading) model. Controls the actual training process by dividing the workload across the available processes.

    Args:
        data_params (dict): Prameters to initialize the datasets.
        architecture_params (dict): Architecture parameters.
        model_params (dict): Model parameters. 
        loader_params (dict): Dataloader parameters.
        optimizer_params (dict): Optimizer parameters.
        training_params (dict): Training parameters. 
    """

    # Make training and testing datasets
    train_dataset, test_dataset = data.CUBE_make_training_and_testing_data(
        **data_params
    )

    # Make model object
    model = model_params["architecture"](**architecture_params)
    
    # Load the state
    state = tutils.get_state(model_params)

    # If not new state, load the current state dict. 
    if state["model_state_dict"] is not None:
        model.load_state_dict(state["model_state_dict"])

    # Define/collect the arguments that will be sent to the worker process
    worker_args = {
        "world_size": world_size,
        "model": model,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "loader_params": loader_params,
        "optimizer_params": optimizer_params,
        "training_params": training_params,
        "state": state,
    }

    # Spawn the different processes. 
    mp.spawn(
        worker,
        args=(tuple(worker_args.values())),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    pass
