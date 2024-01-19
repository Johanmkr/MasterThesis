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

import data

output_func = nn.Sigmoid()


def get_state(model_params: dict):
    state = {
        "epoch": 0,
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "train_loss": 0,
        "test_loss": 0,
        "best_loss": 1e10,
        "model_save_path": model_params["model_save_path"],
    }
    if model_params["load_model"]:
        # Load model
        try:
            state = torch.load(model_params["model_save_path"])
            print(
                f"Loaded model from {model_params['model_save_path']}\nAlready trained for {state['epoch']} epochs"
            )
        except FileNotFoundError:
            print("No model found. Training from scratch.")
    return state


def success(outputs, labels, tol):
    return (abs(output_func(outputs) - labels) < tol).sum().item()


def train_one_epoch(
    device,
    model,
    train_loader,
    optimizer,
    loss_fn,
    epoch_nr,
    success_tol=0.5,
):
    print(f"---------- Epoch {epoch_nr} ----------\n") if (
        device == 0 or type(device) == torch.device
    ) else None
    model.train()
    train_loss = 0
    train_predictions = 0
    train_samples = 0
    max_batches = len(train_loader)

    i = 0
    iterator = iter(train_loader)
    end_of_data = False
    while not end_of_data:
        try:
            batch = next(iterator)
            if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and (
                device == 0 or type(device) == torch.device
            ):
                print(f"TRAIN - Batch: {i+1}/{max_batches}")
            # Get the inputs
            images, labels = batch["image"], batch["label"]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            train_loss += loss.item()
            train_predictions += success(outputs, labels, tol=success_tol)
            train_samples += len(labels)
            i += 1
        except StopIteration:
            end_of_data = True

    train_loss /= max_batches  # avg loss per batch

    return train_loss, train_predictions, train_samples


def evaluate(
    device,
    model,
    loss_fn,
    test_loader,
    success_tol=0.5,
):
    model.eval()
    test_loss = 0
    evaluation_predictions = 0
    evaluation_samples = 0
    max_batches = len(test_loader)

    i = 0
    iterator = iter(test_loader)
    end_of_data = False

    with torch.no_grad():
        while not end_of_data:
            try:
                batch = next(iterator)
                if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and (
                    device == 0 or type(device) == torch.device
                ):
                    print(f"EVAL - Batch: {i+1}/{max_batches}")
                images, labels = batch["image"], batch["label"]
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Print statistics
                test_loss += loss.item()
                evaluation_predictions += success(outputs, labels, tol=success_tol)
                evaluation_samples += len(labels)
            except StopIteration:
                end_of_data = True
    test_loss /= max_batches  # avg loss per batch

    return (
        test_loss,
        evaluation_predictions,
        evaluation_samples,
    )


def print_and_write_statistics(
    writer, epoch_nr, loss, predictions, samples, suffix, time=None
):
    string = f"\n---{suffix.capitalize()}---\nEpoch {epoch_nr}\nLoss: {loss:.5f}\nAccuracy: {predictions/samples*100:.2f} %\n"
    string += f"Time: {time:.2f}\n" if time is not None else ""
    print(string)

    writer.add_scalar(f"Loss/{suffix}", loss, epoch_nr)
    writer.add_scalar(f"Accuracy/{suffix}", predictions / samples, epoch_nr)
    writer.flush()
