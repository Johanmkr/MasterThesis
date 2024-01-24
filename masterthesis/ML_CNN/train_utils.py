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

import data

output_func = nn.Sigmoid()


transforms = [lambda x: x]

rotate90 = tf.RandomRotation((90, 90))
rotate180 = tf.RandomRotation((180, 180))
rotate270 = tf.RandomRotation((270, 270))
flipH = tf.RandomHorizontalFlip(p=1.0)

rot90 = tf.Compose([rotate90])
rot180 = tf.Compose([rotate180])
rot270 = tf.Compose([rotate270])
flip = tf.Compose([flipH])
fliprot90 = tf.Compose([flipH, rotate90])
fliprot180 = tf.Compose([flipH, rotate180])
fliprot270 = tf.Compose([flipH, rotate270])

for transform in [rot90, rot180, rot270, flip, fliprot90, fliprot180, fliprot270]:
    transforms.append(transform)

nr_transformations = len(transforms)


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


def confusion_metrics(predictions, targets, success_tol=0.5, target_noise=0.0):
    target_noise += 1e-6  # Add a small number to avoid rounding errors

    if predictions.size(0) != targets.size(0):
        raise ValueError("The length of predictions and targets must be the same.")

    # Normalize target values to be either 0 or 1 with the given tolerance
    targets = (
        targets.clone()
    )  # Clone to prevent in-place operations affecting the original tensor
    targets[targets < (0 + target_noise)] = 0
    targets[targets > (1 - target_noise)] = 1

    # Apply threshold to predictions
    predictions = (predictions >= success_tol).bool()

    # Convert targets to boolean for comparison
    targets_bool = targets.bool()

    # True Positives (TP): both prediction and target are True (1)
    TP = torch.sum(predictions & targets_bool).item()

    # True Negatives (TN): both prediction and target are False (0)
    TN = torch.sum(~predictions & ~targets_bool).item()

    # False Positives (FP): prediction is True (1) but target is False (0)
    FP = torch.sum(predictions & ~targets_bool).item()

    # False Negatives (FN): prediction is False (0) but target is True (1)
    FN = torch.sum(~predictions & targets_bool).item()

    return TP, TN, FP, FN


def one_pass(model, optimizer, loss_fn, image, labels):
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


# def train_one_epoch(
#     device,
#     model,
#     train_loader,
#     optimizer,
#     loss_fn,
#     epoch_nr,
#     success_tol=0.5,
# ):
#     print(f"---------- Epoch {epoch_nr} ----------\n") if (
#         device == 0 or type(device) == torch.device
#     ) else None
#     model.train()
#     train_loss = 0
#     train_predictions = 0
#     train_samples = 0
#     max_batches = len(train_loader)

#     i = 0
#     iterator = iter(train_loader)
#     end_of_data = False
#     while not end_of_data:
#         try:
#             batch = next(iterator)
#             if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and (
#                 device == 0 or type(device) == torch.device
#             ):
#                 print(f"TRAIN - Batch: {i+1}/{max_batches}")
#             # Get the inputs
#             images, labels = batch["image"], batch["label"]
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward + backward + optimize
#             outputs = model(images)
#             loss = loss_fn(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # Print statistics
#             train_loss += loss.item()
#             train_predictions += success(outputs, labels, tol=success_tol)
#             train_samples += len(labels)
#             i += 1
#         except StopIteration:
#             end_of_data = True

#     train_loss /= max_batches  # avg loss per batch

#     return train_loss, train_predictions, train_samples


# def evaluate(
#     device,
#     model,
#     loss_fn,
#     test_loader,
#     success_tol=0.5,
# ):
#     model.eval()
#     test_loss = 0
#     evaluation_predictions = 0
#     evaluation_samples = 0
#     max_batches = len(test_loader)

#     i = 0
#     iterator = iter(test_loader)
#     end_of_data = False

#     with torch.no_grad():
#         while not end_of_data:
#             try:
#                 batch = next(iterator)
#                 if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and (
#                     device == 0 or type(device) == torch.device
#                 ):
#                     print(f"EVAL - Batch: {i+1}/{max_batches}")
#                 images, labels = batch["image"], batch["label"]
#                 images = images.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True)

#                 loss, outputs = one_eval(model, loss_fn, images, labels)

#                 # Print statistics
#                 test_loss += loss.item()
#                 evaluation_predictions += success(outputs, labels, tol=success_tol)
#                 evaluation_samples += len(labels)
#                 i += 1
#             except StopIteration:
#                 end_of_data = True
#     test_loss /= max_batches  # avg loss per batch

#     return (
#         test_loss,
#         evaluation_predictions,
#         evaluation_samples,
#     )


def print_and_write_statistics(
    writer, epoch_nr, loss, TP, TN, FP, FN, suffix, time=None
):
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * (precision * recall) / (precision + recall)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

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
                if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and (
                    device == 0 or type(device) == torch.device
                ):
                    print(f"EVAL - Batch: {i+1}/{max_batches}")
                cubes, labels = batch["cube"], batch["label"]

                # Send to device
                cubes = cubes.to(device, non_blocking=False)
                labels = labels.to(device, non_blocking=False)

                transforms_idices = list(np.random.permutation(nr_transformations))

                # Stack together the cubes and labels in the batch
                N, D, H, W = cubes.shape

                slices_d = cubes.permute(0, 2, 3, 1).reshape(
                    -1, 1, H, W
                )  # (N*256, 1, 256, 256)
                slices_h = cubes.permute(0, 1, 3, 2).reshape(
                    -1, 1, D, W
                )  # (N*256, 1, 256, 256)
                slices_w = cubes.reshape(-1, 1, D, H)  # (N*256, 1, 256, 256)

                all_slices = torch.cat(
                    [slices_d, slices_h, slices_w], dim=0
                )  # (3*N*256, 1, 256, 256)

                targets = labels.repeat(1, 256 * 3).reshape(-1, 1)  # (3*N*256, 1)

                # Loop over transformations
                for transform_idx in transforms_idices:
                    transformed_slices = transforms[transform_idx](all_slices)
                    loss, pred = one_eval(model, loss_fn, transformed_slices, targets)
                    test_loss += loss.item()
                    TP_, TN_, FP_, FN_ = confusion_metrics(
                        pred,
                        targets,
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
    print(f"---------- Epoch {epoch_nr} ----------\n") if (
        device == 0 or type(device) == torch.device
    ) else None
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
            if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and (
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

            # Stack together the cubes and labels in the batch
            N, D, H, W = cubes.shape

            slices_d = cubes.permute(0, 2, 3, 1).reshape(
                -1, 1, H, W
            )  # (N*256, 1, 256, 256)
            slices_h = cubes.permute(0, 1, 3, 2).reshape(
                -1, 1, D, W
            )  # (N*256, 1, 256, 256)
            slices_w = cubes.reshape(-1, 1, D, H)  # (N*256, 1, 256, 256)

            all_slices = torch.cat(
                [slices_d, slices_h, slices_w], dim=0
            )  # (3*N*256, 1, 256, 256)

            targets = labels.repeat(1, 256 * 3).reshape(-1, 1)  # (3*N*256, 1)

            # Loop over transformations
            for transform_idx in transforms_idices:
                transformed_slices = transforms[transform_idx](all_slices)
                loss, pred = one_pass(
                    model, optimizer, loss_fn, transformed_slices, targets
                )
                train_loss += loss.item()
                TP_, TN_, FP_, FN_ = confusion_metrics(
                    pred,
                    targets,
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
