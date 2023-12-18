import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import time
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

GPU = torch.cuda.is_available()
world_size = torch.cuda.device_count()


class MultipleGPUTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: nn.optim,
        loss_fn: nn.Loss,
        train_dataset,
        test_dataset,
        batch_size,
        num_workers,
        prefetch_factor,
        test_name: str = "test",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.model_save_path = f"models/{test_name}.pt"
        self.writer_log_path = f"runs/{test_name}"

        # Tensorboard
        self.writer = SummaryWriter(self.writer_log_path)
        self.writer.add_graph(self.model, torch.zeros(model.input_size).unsqueeze(0))

    def create_data_loaders(
        self,
        rank,
        world_size,
        train_dataset,
        test_dataset,
        batch_size,
        num_workers=0,
        prefetch_factor=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
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
        return train_loader, test_loader

    def _setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def _cleanup(self):
        dist.destroy_process_group()

    def _success(self, outputs, labels, tol):
        return (abs(outputs - labels) < tol).sum().item()

    def main_ddp_init(self, rank, world_size, model):
        # setup the process group
        self._setup(rank, world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(42)

        # create model and move it to GPU with id rank
        model = model.to(rank)
        ddp_model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )

        return ddp_model

    def train_one_epoch(
        self,
        rank,
        model,
        train_loader,
        optimizer,
        loss_fn,
        epoch_nr,
        success_tol=1e-2,
    ):
        epoch_train_start_time = time.time()
        print(f"---------- Epoch {epoch_nr} ----------\n")
        model.train()
        train_loss = 0
        train_predictions = 0
        train_samples = 0
        max_batches = len(train_loader)
        for i, data in enumerate(train_loader):
            if (i + 1) % 25 == 0 and rank == 0:
                print(f"Batch: {i+1}/{max_batches}")
            # Get the inputs
            images, labels = data["image"], data["label"]
            images = images.to(rank)
            labels = labels.to(rank)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            train_loss += loss.item()
            train_predictions += self._success(outputs, labels, tol=success_tol)
            train_samples += len(labels)
        train_loss /= max_batches  # avg loss per batch
        epoch_train_end_time = time.time()
        print(
            f"\nTraining:\nTrain loss: {train_loss:.4f}\nTrain predictions: {train_predictions}/{train_samples}\nTrain accuracy: {train_predictions/train_samples*100:.4f} %\nTime elapsed for training: {epoch_train_end_time - epoch_train_start_time:.2f} s\n"
        ) if rank == 0 else None
        return train_loss, train_predictions, train_samples

    def evaluate(
        model,
        rank,
        loss_fn,
        test_loader,
        success_tol=0.5,
    ):
        epoch_evaluation_start_time = time.time()
        model.eval()
        test_loss = 0
        evaluation_predictions = 0
        evaluation_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["image"], batch["label"]
                images = images.to(rank)
                labels = labels.to(rank)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Print statistics
                test_loss += loss.item()
                evaluation_predictions += self._success(
                    outputs, labels, tol=success_tol
                )
                evaluation_samples += len(labels)
        epoch_evaluation_end_time = time.time()
        print(
            f"Testing:\nTest loss: {test_loss:.4f}\nTest predictions: {evaluation_predictions}/{evaluation_samples}\nTest accuracy: {evaluation_predictions/evaluation_samples*100:.4f} %\nTime elapsed for testing: {epoch_evaluation_end_time - epoch_evaluation_start_time:.2f} s\n"
        ) if rank == 0 else None
        return test_loss, evaluation_predictions, evaluation_samples
