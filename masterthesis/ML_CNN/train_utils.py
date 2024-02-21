TRANSFORMS = False

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
from torchvision import transforms as tf
import matplotlib.pyplot as plt

import data

# Set output function
output_func = nn.Sigmoid()

# Initialise transform list. First should be just identity transform.
transforms = [lambda x: x]

# Create the different uniqe transformation.
rotate90 = tf.RandomRotation((90, 90))
rotate180 = tf.RandomRotation((180, 180))
rotate270 = tf.RandomRotation((270, 270))
flipH = tf.RandomHorizontalFlip(p=1.0)

# Make compositions.
rot90 = tf.Compose([rotate90])
rot180 = tf.Compose([rotate180])
rot270 = tf.Compose([rotate270])
flip = tf.Compose([flipH])
fliprot90 = tf.Compose([flipH, rotate90])
fliprot180 = tf.Compose([flipH, rotate180])
fliprot270 = tf.Compose([flipH, rotate270])

if TRANSFORMS:
    # Append all transformations to the list
    for transform in [rot90, rot180, rot270, flip, fliprot90, fliprot180, fliprot270]:
        transforms.append(transform)

# Keep track of the number of transformations.
nr_transformations = len(transforms)


def get_state(model_params: dict) -> dict:
    """Initalize a new state dictionary. Tries to fill it if load is allowed and the file exists, else traines from scratch with the new state.

    Args:
        model_params (dict): Model parameters.

    Returns:
        dict: State dictionary.
    """

    # New state dict
    state = {
        "epoch": 0,
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "train_loss": 0,
        "test_loss": 0,
        "best_loss": 1e10,
        "model_save_path": model_params["model_save_path"],
        "model_information_written": False,
    }

    # Tries to load if allowed (and it exists)
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


def confusion_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, success_tol: float = 0.5
) -> tuple:
    """Calculates the true positive (TN), true negative (TN), false positive (FP) and false negative (FN) values given a tensor fo predictions and targets.

    Args:
        predictions (torch.Tensor): Predictions from the model (after sigmoid output).
        targets (torch.Tensro): Target labels (0 or 1).
        success_tol (float, optional): Success tolerance. Defaults to 0.5.

    Raises:
        ValueError: If the two tensors have unequal shape.

    Returns:
        tuple: Tuple with calculated values: (TP, TN, FP, FN)
    """
    if predictions.size(0) != targets.size(0):
        raise ValueError("The length of predictions and targets must be the same.")

    # Apply threshold to predictions
    predictions = (predictions >= success_tol).bool()

    # Convert targets to bool
    targets_bool = targets.bool()

    # Calculate metrics
    TP = torch.sum(predictions & targets_bool).item()
    TN = torch.sum(~predictions & ~targets_bool).item()
    FP = torch.sum(predictions & ~targets_bool).item()
    FN = torch.sum(~predictions & targets_bool).item()

    return TP, TN, FP, FN


def create_confusion_matrix(TP, TN, FP, FN, normalize=False):
    # Matplotlib figure of confusion matrix
    ###FIXME this does not work as intended
    fig, ax = plt.subplots()
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (NG)", "1 (GR)"])
    ax.set_yticklabels(["0 (NG)", "1 (GR)"])
    total_positive = TP + FN
    total_negative = TN + FP
    if normalize:
        cf = [[TN / (TN + FP), FP / (TN + FP)], [FN / (FN + TP), TP / (FN + TP)]]

    else:
        cf = [[TN, FP], [FN, TP]]
    # Add text to each cell
    for i in range(2):
        for j in range(2):
            f = "{:.2f}" if normalize else "{:d}"
            ax.text(i, j, f"{cf[i][j]:.2f}", ha="center", va="center")
    img = ax.imshow(cf, cmap="Blues")
    return_image = fig.get_figure()
    fig.close()
    return return_image


def one_pass(
    model: torch.nn.Module,
    optimizer: any,
    loss_fn: any,
    image: torch.Tensor,
    labels: torch.Tensor,
) -> tuple:
    optimizer.zero_grad()
    output = model(image)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    return loss, output_func(output)


def one_eval(model, loss_fn, image, labels):
    output = model(image)
    loss = loss_fn(output, labels)
    return loss, output_func(output)


def slice_and_rearrange_cube(cubes, labels, device):
    # Stack together the cubes and labels in the batch
    N, D, H, W = cubes.shape

    slices = []
    targets = []
    for i in range(N):
        slice_d = (
            cubes[i, :, :, :].unsqueeze(0).permute(0, 2, 3, 1).reshape(-1, 1, H, W)
        )  # (256, 1, 256, 256)
        slice_h = (
            cubes[i, :, :, :].unsqueeze(0).permute(0, 1, 3, 2).reshape(-1, 1, D, W)
        )  # (256, 1, 256, 256)
        slice_w = (
            cubes[i, :, :, :].unsqueeze(0).reshape(-1, 1, D, H)
        )  # (256, 1, 256, 256)
        slices.append(slice_d)
        slices.append(slice_h)
        slices.append(slice_w)

        target = torch.ones(256, 1).to(device) * labels[i]
        targets.append(target)
        targets.append(target)
        targets.append(target)

    all_slices = torch.cat(slices, dim=0)
    all_targets = torch.cat(targets, dim=0)

    # Old version

    # slices_d = cubes.permute(0, 2, 3, 1).reshape(-1, 1, H, W)  # (N*256, 1, 256, 256)
    # slices_h = cubes.permute(0, 1, 3, 2).reshape(-1, 1, D, W)  # (N*256, 1, 256, 256)
    # slices_w = cubes.reshape(-1, 1, D, H)  # (N*256, 1, 256, 256)

    # all_slices = torch.cat(
    #     [slices_d, slices_h, slices_w], dim=0
    # )  # (3*N*256, 1, 256, 256)

    # all_targets = labels.repeat(1, 256 * 3).reshape(-1, 1)  # (3*N*256, 1)

    return all_slices, all_targets


def calculate_metrics(TP, TN, FP, FN):
    # Calculate metrics
    divtol = 1e-9
    accuracy = (TP + TN) / (TP + TN + FP + FN + divtol)
    precision = TP / (TP + FP + divtol)
    recall = TP / (TP + FN + divtol)
    F1_score = 2 * (precision * recall) / (precision + recall + divtol)
    TPR = TP / (TP + FN + divtol)
    FPR = FP / (FP + TN + divtol)
    return accuracy, precision, recall, F1_score, TPR, FPR


def print_and_write_statistics(
    writer, epoch_nr, loss, TP, TN, FP, FN, suffix, time=None
):
    accuracy, precision, recall, F1_score, TPR, FPR = calculate_metrics(TP, TN, FP, FN)

    # Print statistics
    string = f"\n---{suffix.capitalize()}---\nEpoch {epoch_nr}\nLoss:       {loss:.5f}\nAccuracy:   {accuracy:.4f} = {accuracy*100:.2f} %\nPrecision:  {precision:.4f}\nRecall:     {recall:.4f}\nF1 score:   {F1_score:.4f}\nTPR:        {TPR:.4f}\nFPR:        {FPR:.4f}\n"
    string += f"Time: {time:.2f}\n" if time is not None else ""
    print(string)

    for quantity in [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "F1_score",
        "TPR",
        "FPR",
    ]:
        writer.add_scalar(f"{quantity}/{suffix}", eval(quantity), epoch_nr)
    writer.flush()

    # writer.add_scalar(f"Loss/{suffix}", loss, epoch_nr)
    # writer.add_scalar(f"Accuracy/{suffix}", predictions / samples, epoch_nr)
    # writer.flush()


def evaluate_cube_version(
    device,
    model,
    loss_fn,
    test_loader,
):
    model.eval()
    test_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    max_batches = len(test_loader)

    i = 0
    iterator = iter(test_loader)
    end_of_data = False

    with torch.no_grad():
        while not end_of_data:
            try:
                batch = next(iterator)
                if ((i + 1) % 10 == 0 or (i + 1) == max_batches) and (
                    device == 0 or type(device) == torch.device
                ):
                    print(f"EVAL - Batch: {i+1}/{max_batches}")
                cubes, labels = batch["cube"], batch["label"]

                # Send to device
                cubes = cubes.to(device, non_blocking=False)
                labels = labels.to(device, non_blocking=False)

                transforms_idices = list(np.random.permutation(nr_transformations))

                all_slices, targets = slice_and_rearrange_cube(cubes, labels, device)

                # Loop over transformations
                for transform_idx in transforms_idices:
                    # Create a permutation index that will be used to shuffle both the slices and the labels.
                    permutation_index = torch.randperm(all_slices.size(0))

                    loss, pred = one_eval(
                        model,
                        loss_fn,
                        transforms[transform_idx](all_slices[permutation_index]),
                        targets[permutation_index],
                    )
                    test_loss += loss.item()
                    TP_, TN_, FP_, FN_ = confusion_metrics(
                        pred,
                        targets[permutation_index],
                    )
                    TP += TP_
                    TN += TN_
                    FP += FP_
                    FN += FN_
                i += 1
            except StopIteration:
                end_of_data = True
    test_loss /= max_batches * nr_transformations  # avg loss per batch

    return (
        test_loss,
        TP,
        TN,
        FP,
        FN,
    )


# input is now a whole datacube
def train_one_epoch_cube_version(
    device,
    model,
    train_loader,
    optimizer,
    loss_fn,
    epoch_nr,
):
    (
        print(f"---------- Epoch {epoch_nr} ----------\n")
        if (device == 0 or type(device) == torch.device)
        else None
    )
    model.train()
    train_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    max_batches = len(train_loader)

    i = 0
    added_loss = 0
    iterator = iter(train_loader)
    end_of_data = False
    while not end_of_data:
        try:
            batch = next(iterator)
            if ((i + 1) % 10 == 0 or (i + 1) == max_batches) and (
                device == 0 or type(device) == torch.device
            ):
                print(f"TRAIN - Batch: {i+1}/{max_batches}")
            # Get the inputs
            # Shape of cubes will be (batch_size, 1, 256, 256, 256)
            # Shape of labels will be (batch_size, 1)
            cubes, labels = batch["cube"], batch["label"]

            # Send to device
            cubes = cubes.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            transforms_idices = list(np.random.permutation(nr_transformations))

            all_slices, targets = slice_and_rearrange_cube(cubes, labels, device)

            # Loop over transformations
            for transform_idx in transforms_idices:
                # Create a permutation index that will be used to shuffle both the slices and the labels.
                permutation_index = torch.randperm(all_slices.size(0))

                loss, pred = one_pass(
                    model,
                    optimizer,
                    loss_fn,
                    transforms[transform_idx](all_slices[permutation_index]),
                    targets[permutation_index],
                )
                train_loss += loss.item()
                TP_, TN_, FP_, FN_ = confusion_metrics(
                    pred,
                    targets[permutation_index],
                )
                TP += TP_
                TN += TN_
                FP += FP_
                FN += FN_

            i += 1
        except StopIteration:
            end_of_data = True

    train_loss /= max_batches * nr_transformations  # avg loss per batch

    return (
        train_loss,
        TP,
        TN,
        FP,
        FN,
    )


def train_one_epoch(
    device,
    model,
    train_loader,
    optimizer,
    loss_fn,
    epoch_nr,
):
    (
        print(f"---------- Epoch {epoch_nr} ----------\n")
        if (device == 0 or type(device) == torch.device)
        else None
    )
    model.train()
    train_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    max_batches = len(train_loader)

    i = 0
    added_loss = 0
    iterator = iter(train_loader)
    end_of_data = False
    while not end_of_data:
        try:
            batch = next(iterator)
            if ((i + 1) % 10 == 0 or (i + 1) == max_batches) and (
                device == 0 or type(device) == torch.device
            ):
                print(f"TRAIN - Batch: {i+1}/{max_batches}")

            images, labels = batch["image"], batch["label"]

            # Send to device
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            loss, pred = one_pass(
                model,
                optimizer,
                loss_fn,
                images,
                labels,
            )
            train_loss += loss.item()
            TP_, TN_, FP_, FN_ = confusion_metrics(
                pred,
                labels,
            )
            TP += TP_
            TN += TN_
            FP += FP_
            FN += FN_

            i += 1
        except StopIteration:
            end_of_data = True

    train_loss /= max_batches  # avg loss per batch

    return (
        train_loss,
        TP,
        TN,
        FP,
        FN,
    )


def evaluate(
    device,
    model,
    loss_fn,
    test_loader,
):
    model.eval()
    test_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    max_batches = len(test_loader)

    i = 0
    iterator = iter(test_loader)
    end_of_data = False

    with torch.no_grad():
        while not end_of_data:
            try:
                batch = next(iterator)
                if ((i + 1) % 10 == 0 or (i + 1) == max_batches) and (
                    device == 0 or type(device) == torch.device
                ):
                    print(f"EVAL - Batch: {i+1}/{max_batches}")
                images, labels = batch["image"], batch["label"]

                # Send to device
                images = images.to(device, non_blocking=False)
                labels = labels.to(device, non_blocking=False)

                loss, pred = one_eval(
                    model,
                    loss_fn,
                    images,
                    labels,
                )
                test_loss += loss.item()
                TP_, TN_, FP_, FN_ = confusion_metrics(
                    pred,
                    labels,
                )
                TP += TP_
                TN += TN_
                FP += FP_
                FN += FN_
                i += 1
            except StopIteration:
                end_of_data = True
    test_loss /= max_batches  # avg loss per batch

    return (
        test_loss,
        TP,
        TN,
        FP,
        FN,
    )
