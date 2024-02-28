import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import time
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import data_image as data
import train_utils as tutils


def create_data_loaders(
    train_dataset,
    test_dataset,
    batch_size,
    num_workers=0,
    prefetch_factor=2,
    pin_memory=False,
):
    # Create distributed samplers
    train_sampler = RandomSampler(
        train_dataset,
    )
    test_sampler = RandomSampler(
        test_dataset,
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


def where_stuff_happens(
    device,
    model,
    train_dataset,
    test_dataset,
    loader_params,
    optimizer_params,
    training_params,
    state,
):
    # Data loaders
    train_loader, test_loader, train_sampler, test_sampler = create_data_loaders(
        train_dataset,
        test_dataset,
        **loader_params,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    if state["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    loss_fn = nn.BCEWithLogitsLoss()

    epochs_trained = state["epoch"]

    # Tensorboard
    writer = SummaryWriter(training_params["writer_log_path"])

    # Train and test
    try:
        best_loss = state["best_loss"]
    except KeyError:
        best_loss = 1e10
    for epoch in range(
        epochs_trained + 1, epochs_trained + training_params["epochs"] + 1
    ):
        epoch_total_time_start = time.time()

        # ---- TRAINING ----
        # train_sampler.set_epoch(epoch)
        epoch_train_start_time = time.time()
        (
            train_loss,
            train_TP,
            train_TN,
            train_FP,
            train_FN,
        ) = tutils.train_one_epoch(
            device=device,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch_nr=epoch,
        )
        epoch_train_end_time = time.time()

        tutils.print_and_write_statistics(
            writer=writer,
            epoch_nr=epoch,
            loss=train_loss,
            TP=train_TP,
            TN=train_TN,
            FP=train_FP,
            FN=train_FN,
            suffix="train",
            time=epoch_train_end_time - epoch_train_start_time,
        )

        if epoch % training_params["test_every"] == 0:
            # ---- TESTING ----
            epoch_test_start_time = time.time()
            (
                test_loss,
                test_TP,
                test_TN,
                test_FP,
                test_FN,
            ) = tutils.evaluate(
                device=device,
                model=model,
                test_loader=test_loader,
                loss_fn=loss_fn,
            )
            epoch_test_end_time = time.time()
            tutils.print_and_write_statistics(
                writer=writer,
                epoch_nr=epoch,
                loss=test_loss,
                TP=test_TP,
                TN=test_TN,
                FP=test_FP,
                FN=test_FN,
                suffix="test",
                time=epoch_test_end_time - epoch_test_start_time,
            )

            # Save one model with the current epoch (copy) and overwrite the master save
            best_loss = test_loss if test_loss < best_loss else best_loss
            state["epoch"] = epoch
            state["model_state_dict"] = model.state_dict()
            state["optimizer_state_dict"] = optimizer.state_dict()
            state["train_loss"] = train_loss
            state["test_loss"] = test_loss
            state["best_loss"] = best_loss
            # from IPython import embed

            # embed()
            epoch_savepath = (
                "/".join(state[f"model_save_path"].split("/")[:-1])
                + "/"
                + "TMP_"
                + state[f"model_save_path"].split("/")[-1].replace(".pt", "")
                + f"_epoch{state['epoch']}.pt"
            )

            # Save a new copy of the model for the current epoch
            (
                torch.save(
                    state,
                    epoch_savepath,
                )
                if training_params["save_tmp_every"] % epoch == 0
                else None
            )

            # Save/overwrite the current master model
            torch.save(
                state,
                state["model_save_path"],
            )
            print(f"Saved model to {epoch_savepath}")

        epoch_total_time_end = time.time()
        print(
            f"Time elapsed for epoch: {epoch_total_time_end - epoch_total_time_start:.2f} s\n"
        )

        writer.flush()
        writer.close()


def train(
    data_params,
    architecture_params,
    model_params,
    loader_params,
    optimizer_params,
    training_params,
):
    train_dataset, test_dataset = data.make_training_and_testing_data(**data_params)

    model = model_params["architecture"](**architecture_params)
    state = tutils.get_state(model_params)
    if state["model_state_dict"] is not None:
        model.load_state_dict(state["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    where_stuff_happens(
        device,
        model,
        train_dataset,
        test_dataset,
        loader_params,
        optimizer_params,
        training_params,
        state,
    )


if __name__ == "__main__":
    pass
