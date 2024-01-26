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
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def model_init(rank, world_size, model):
    # setup the process group
    setup(rank, world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

    # create model and move it to GPU with id rank
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    return ddp_model


def cleanup():
    dist.destroy_process_group()


##################################################################


def create_data_loaders(
    rank,
    world_size,
    train_dataset,
    test_dataset,
    batch_size,
    num_workers=0,
    prefetch_factor=2,
    pin_memory=False,
    shuffle=True,
    drop_last=False,
):
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
    rank,
    world_size,
    model,
    train_dataset,
    test_dataset,
    loader_params,
    optimizer_params,
    training_params,
    state,
):
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
    if state["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    loss_fn = nn.BCEWithLogitsLoss()

    epochs_trained = state["epoch"]

    # Tensorboard
    if rank == 0:
        writer = SummaryWriter(training_params["writer_log_path"])
        # writer.add_graph(ddp_model, torch.zeros((1, 256, 256)).unsqueeze(0))

    # Train and test
    try:
        best_loss = state["best_loss"]
    except KeyError:
        best_loss = 1e10
    for epoch in range(
        epochs_trained + 1, epochs_trained + training_params["epochs"] + 1
    ):
        epoch_total_time_start = time.time() if rank == 0 else None

        # ---- TRAINING ----
        train_sampler.set_epoch(epoch)
        epoch_train_start_time = time.time() if rank == 0 else None
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
        reduced_train_metrics = [
            torch.zeros_like(train_metrics) for _ in range(world_size)
        ]

        dist.all_gather(reduced_train_metrics, train_metrics)

        # Calculate and log statistics on master rank
        if rank == 0:
            # Train data
            # print(len(reduced_train_metrics))
            (
                total_mean_train_loss,
                total_train_TP,
                total_train_TN,
                total_train_FP,
                total_train_FN,
            ) = (torch.stack(reduced_train_metrics).sum(dim=0)).tolist()
            mean_train_loss = total_mean_train_loss / world_size
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

        # Placeholder for early stopping across all ranks
        should_stop = torch.tensor(False, dtype=torch.bool).to(rank)

        if epoch % training_params["test_every"] == 0:
            # ---- TESTING ----
            epoch_test_start_time = time.time() if rank == 0 else None
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

            # Gather statistics and send to rank 0
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
                # Test data
                (
                    total_mean_test_loss,
                    total_test_TP,
                    total_test_TN,
                    total_test_FP,
                    total_test_FN,
                ) = (torch.stack(reduced_test_metrics).sum(dim=0)).tolist()
                mean_test_loss = total_mean_test_loss / world_size

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

                # Early stopping condition
                if mean_test_loss < training_params["breakout_loss"]:
                    print(
                        f"Breaking out of training loop because test loss {mean_test_loss:.4f} < breakout loss {training_params['breakout_loss']:.4f}"
                    )
                    should_stop = torch.tensor(True, dtype=torch.bool).to(rank)

                # Save model if mean loss is better than best loss of if early stopping condition is met
                if should_stop or mean_test_loss < best_loss:
                    best_loss = mean_test_loss
                    state["epoch"] = epoch
                    state["model_state_dict"] = model.state_dict()
                    state["optimizer_state_dict"] = optimizer.state_dict()
                    state["train_loss"] = mean_train_loss
                    state["test_loss"] = mean_test_loss
                    state["best_loss"] = best_loss

                    torch.save(
                        state,
                        state["model_save_path"],
                    )
                    print(
                        f"New best loss: {best_loss:.4f}. Saved model to {state['model_save_path']}"
                    )
        if rank == 0:
            epoch_total_time_end = time.time()

            print(
                f"Time elapsed for epoch: {epoch_total_time_end - epoch_total_time_start:.2f} s\n"
            )
        # Broadcast early stopping condition to all ranks
        dist.broadcast(should_stop, src=0)

        # Break out of training loop if early stopping condition is met
        if should_stop:
            print(f"Rank {rank} breaking out of training loop")
            break

        # Synchronize all processes
        dist.barrier()

    print(f"Trained for {epoch} epochs") if rank == 0 else None
    print(f"Rank {rank} finished training")

    # Closing writer and cleaning up
    if rank == 0:
        writer.flush()
        writer.close()

    cleanup()


def train(
    data_params,
    architecture_params,
    model_params,
    loader_params,
    optimizer_params,
    training_params,
):
    train_dataset, test_dataset = data.CUBE_make_training_and_testing_data(
        **data_params
    )

    model = model_params["architecture"](**architecture_params)
    state = tutils.get_state(model_params)
    if state["model_state_dict"] is not None:
        model.load_state_dict(state["model_state_dict"])

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
    mp.spawn(
        worker,
        args=(tuple(worker_args.values())),
        # args=(world_size, self.train_dataset, self.test_dataset, self.model),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    pass
