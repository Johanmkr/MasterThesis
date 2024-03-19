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

if __name__ == "__main__":
    import data
else:
    from .  import data
# import data if __name__ == "__main__" else import .data

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


def get_state(model_path, load_model=True, device="cuda") -> dict:
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
        "model_save_path": model_path,
        "model_information_written": False,
        "tmp_save_count": 0,
    }

    # Tries to load if allowed (and it exists)
    if load_model and os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        print(
            f"Loaded model from {model_path}\nAlready trained for {state['epoch']} epochs"
        )
    else:
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


def infer(
    device,
    model,
    loss_fn,
    inf_loader,
):
    model.eval()
    test_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    max_batches = len(inf_loader)

    i = 0
    iterator = iter(inf_loader)
    end_of_data = False
    total_prediction = []
    total_labels = []

    with torch.no_grad():
        while not end_of_data:
            try:
                batch = next(iterator)
                if ((i + 1) % 10 == 0 or (i + 1) == max_batches) and (
                    device == 0 or type(device) == torch.device
                ):
                    print(f"INFER - Batch: {i+1}/{max_batches}")
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
                total_prediction.append(pred)
                total_labels.append(labels)
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
        torch.cat(total_prediction),
        torch.cat(total_labels),
    )
