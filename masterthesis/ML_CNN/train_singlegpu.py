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

import data
import train_utils as tutils


def create_data_loaders(
    train_dataset,
    test_dataset,
    batch_size,
    num_workers=0,
    prefetch_factor=2,
    pin_memory=False,
    shuffle=True,
    drop_last=True,
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
            train_predictions,
            train_samples,
        ) = tutils.train_one_epoch_cube_version(
            device=device,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch_nr=epoch,
            success_tol=training_params["tol"],
        )
        epoch_train_end_time = time.time()

        tutils.print_and_write_statistics(
            writer=writer,
            epoch_nr=epoch,
            loss=train_loss,
            predictions=train_predictions,
            samples=train_samples,
            suffix="train",
            time=epoch_train_end_time - epoch_train_start_time,
        )
        should_stop = torch.tensor(False, dtype=torch.bool).to(device)

        if epoch % training_params["test_every"] == 0:
            # ---- TESTING ----
            epoch_test_start_time = time.time()
            test_loss, test_predictions, test_samples = tutils.evaluate_cube_version(
                device=device,
                model=model,
                test_loader=test_loader,
                loss_fn=loss_fn,
                success_tol=training_params["tol"],
            )
            epoch_test_end_time = time.time()
            tutils.print_and_write_statistics(
                writer=writer,
                epoch_nr=epoch,
                loss=test_loss,
                predictions=test_predictions,
                samples=test_samples,
                suffix="test",
                time=epoch_test_end_time - epoch_test_start_time,
            )

            if test_loss < training_params["breakout_loss"]:
                print(
                    f"Breaking out of training loop because test loss {test_loss:.4f} < breakout loss {training_params['breakout_loss']:.4f}"
                )
                should_stop = torch.tensor(True, dtype=torch.bool).to(device)

            if should_stop or test_loss < best_loss:
                best_loss = test_loss
                state["epoch"] = epoch
                state["model_state_dict"] = model.state_dict()
                state["optimizer_state_dict"] = optimizer.state_dict()
                state["train_loss"] = train_loss
                state["test_loss"] = test_loss
                state["best_loss"] = best_loss

                torch.save(state, state["model_save_path"])

                print(
                    f"New best loss: {best_loss:.4f}. Saved model to {state['model_save_path']}"
                )

        epoch_total_time_end = time.time()
        print(
            f"Time elapsed for epoch: {epoch_total_time_end - epoch_total_time_start:.2f} s\n"
        )

        if should_stop:
            print("Breaking out of training loop because of breakout loss.")
            break

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
    train_dataset, test_dataset = data.CUBE_make_training_and_testing_data(
        **data_params
    )

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
